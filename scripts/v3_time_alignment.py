"""T1: paired physical H4 fine-tuning, uniform versus mix_tmax, with KD off."""
import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import shutil
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.v3_distill import batch_digest, evaluate, state_digest
from scripts.v3_head_sweep import image_seed, sha256


class DiffusionSourceAudit:
    """Hash uncorrupted padded GT and epsilon; x_t is intentionally different."""

    def reset(self):
        self.digest = hashlib.sha256()
        self.count = 0

    def __call__(self, x_start, noise, t):
        self.count += 1
        for tensor in (x_start, noise):
            array = tensor.detach().cpu().contiguous().numpy()
            self.digest.update(str((array.shape, array.dtype)).encode())
            self.digest.update(array.tobytes())


def rng_digest():
    import numpy as np
    import torch
    h = hashlib.sha256()
    h.update(pickle.dumps((random.getstate(), np.random.get_state())))
    for state in (torch.get_rng_state(), torch.cuda.get_rng_state()):
        h.update(state.numpy().tobytes())
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config-file", default="configs/lab/v3/sdd.res50.student4.yaml")
    p.add_argument("--weights", default="output/lab/sdd.res50.bs8/model_final.pth")
    p.add_argument("--dataset-root", default="datasets")
    p.add_argument("--gpu", default="1")
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--train-seeds", type=int, nargs="+", default=[10, 11, 12])
    p.add_argument("--eval-seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--high-probability", type=float, default=0.5)
    p.add_argument("--eval-limit", type=int, default=0)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()
    if args.steps < 1 or args.eval_limit < 0 or args.eval_limit == 1:
        p.error("steps must be positive; eval-limit must be zero or at least two")
    if not 0 <= args.high_probability <= 1:
        p.error("high-probability must be in [0, 1]")
    if any(len(seeds) != len(set(seeds)) for seeds in (args.train_seeds, args.eval_seeds)):
        p.error("Seeds must be unique")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    import numpy as np
    import torch
    import detectron2
    from detectron2.config import get_cfg
    from detectron2.data import DatasetCatalog, build_detection_train_loader
    from detectron2.data.samplers import TrainingSampler
    from detectron2.modeling import build_model
    from detectron2.utils.events import EventStorage
    from detectron2.utils.logger import setup_logger
    from diffusiondet import add_diffusiondet_config, DiffusionDetDatasetMapper
    from diffusiondet.data_register import register_all
    from diffusiondet.distillation import load_truncated_student
    from diffusiondet.time_sampling import PairedTimeSampler
    from diffusiondet.util.model_ema import add_model_ema_configs

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=False)
    setup_logger(output=str(out))
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    def seed_all(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    cfg = get_cfg()
    add_diffusiondet_config(cfg)
    add_model_ema_configs(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.DATASET_ROOT = str(Path(args.dataset_root).resolve())
    cfg.MODEL.DEVICE = "cuda"
    cfg.SOLVER.MAX_ITER = args.steps
    cfg.SOLVER.STEPS = (max(1, int(args.steps * 0.8)),)
    cfg.SOLVER.WARMUP_ITERS = min(cfg.SOLVER.WARMUP_ITERS, max(1, args.steps // 10))
    cfg.SOLVER.WARMUP_FACTOR = 0.1  # Record the factor used by lr_factor below.
    cfg.MODEL.DiffusionDet.TRAIN_T_DIST = "uniform"
    cfg.MODEL.DiffusionDet.TRAIN_T_HIGH_PROB = args.high_probability
    cfg.OUTPUT_DIR = str(out)
    cfg.freeze()
    dd = cfg.MODEL.DiffusionDet
    if dd.NUM_HEADS != 4 or dd.FORMULATION != "vp" or dd.SAMPLE_STEP != 1:
        raise ValueError("T1 requires a physical four-stage, NFE=1 VP student")
    if dd.BOX_RENEWAL or dd.USE_ENSEMBLE or dd.ATTENTION_NEED_WEIGHTS:
        raise ValueError("T1 requires renewal/ensemble/attention weights disabled")
    if cfg.SOLVER.AMP.ENABLED or cfg.MODEL_EMA.ENABLED:
        raise ValueError("T1 uses FP32 and raw student weights only")
    register_all(cfg=cfg)
    train_records = sorted(DatasetCatalog.get("sdd_train"), key=lambda r: r["image_id"])
    full_val = sorted(DatasetCatalog.get("sdd_val"), key=lambda r: r["image_id"])
    val_records = full_val[:args.eval_limit] if args.eval_limit else full_val
    state = torch.load(args.weights, map_location="cpu")["model"]
    (out / "config.yaml").write_text(cfg.dump())
    sources = [Path(__file__).resolve().relative_to(Path.cwd()), Path(args.config_file),
               *[Path("diffusiondet") / name for name in
                 ("time_sampling.py", "distillation.py", "detector.py", "loss.py", "head.py",
                  "config.py", "dataset_mapper.py")],
               Path("scripts/v3_distill.py"), Path("scripts/v3_head_sweep.py"),
               Path("docs/v3/TIME_ALIGNMENT.md")]
    for source in sources:
        target = out / "source_snapshot" / source
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    manifest = dict(args=vars(args), source_sha256={str(s): sha256(s) for s in sources},
                    checkpoint_sha256=sha256(args.weights), checkpoint=str(Path(args.weights).resolve()),
                    git_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                    torch=torch.__version__, detectron2=detectron2.__version__, cuda=torch.version.cuda,
                    gpu=torch.cuda.get_device_name(), full_val_images=len(full_val),
                    train_images=len(train_records), eval_image_ids=[r["image_id"] for r in val_records],
                    arm_order={str(s): (["uniform", "mix_tmax"] if s % 2 == 0 else ["mix_tmax", "uniform"])
                               for s in args.train_seeds},
                    protocol="FP32; TF32 off; cuDNN deterministic in training, false in eval; fresh AdamW; H4; KD off; final checkpoint only",
                    training_seed_protocol="image_seed(train_seed, iteration) resets Python/NumPy/torch before batch; private CPU mixture gate seeded by image_seed(train_seed, 'tgate:{iteration}'); both arms consume the same device randint",
                    pairing_checks=["initial state", "augmented image/GT", "x_start including padding", "epsilon",
                                    "uniform base t", "mixture gate", "global RNG state after forward"],
                    annotation_sha256={split: sha256(Path(args.dataset_root) / "Strawberry/annotations" /
                                                     f"instances_{split}2017.json") for split in ("train", "val")})
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    experiment_rows = []
    for train_seed in args.train_seeds:
        reference_trace, reference_init = None, None
        for arm in manifest["arm_order"][str(train_seed)]:
            arm_dir = out / f"seed{train_seed}_{arm}"
            arm_dir.mkdir()
            arm_cfg = cfg.clone()
            arm_cfg.defrost()
            arm_cfg.MODEL.DiffusionDet.TRAIN_T_DIST = arm
            arm_cfg.OUTPUT_DIR = str(arm_dir)
            arm_cfg.freeze()
            (arm_dir / "config.yaml").write_text(arm_cfg.dump())
            seed_all(train_seed)
            student = build_model(arm_cfg)
            dropped = load_truncated_student(student, state)
            initial_digest = state_digest(student)
            if reference_init is not None and initial_digest != reference_init:
                raise RuntimeError("Paired initial states differ")
            reference_init = initial_digest
            sampler = PairedTimeSampler()
            audit = DiffusionSourceAudit()
            student.train_time_sampler = sampler
            student.train_diffusion_observer = audit
            seed_all(train_seed)
            loader = iter(build_detection_train_loader(
                dataset=train_records, mapper=DiffusionDetDatasetMapper(arm_cfg, is_train=True),
                sampler=TrainingSampler(len(train_records), seed=train_seed),
                total_batch_size=cfg.SOLVER.IMS_PER_BATCH, aspect_ratio_grouping=True, num_workers=0))
            params = [p for p in student.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(params, lr=cfg.SOLVER.BASE_LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY)

            def lr_factor(step):
                warm = min(1.0, 0.1 + 0.9 * step / max(1, cfg.SOLVER.WARMUP_ITERS))
                return warm * (0.1 if step >= cfg.SOLVER.STEPS[0] else 1.0)

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
            trace, input_trace, t_counts = [], [], Counter()
            student.train()
            torch.cuda.reset_peak_memory_stats()
            tick = time.perf_counter()
            with (arm_dir / "training.jsonl").open("w") as log, EventStorage() as storage:
                for step in range(args.steps):
                    seed_all(image_seed(train_seed, step))
                    sampler.reset(image_seed(train_seed, f"tgate:{step}"))
                    audit.reset()
                    step_start = time.perf_counter()
                    batch = next(loader)
                    digest = batch_digest(batch)
                    optimizer.zero_grad(set_to_none=True)
                    losses = student(batch)
                    if audit.count != len(batch) or len(sampler.timesteps) != len(batch):
                        raise RuntimeError("Incomplete diffusion audit")
                    pair_record = dict(input_digest=digest, source_digest=audit.digest.hexdigest(),
                                       base_timesteps=sampler.base_timesteps, gates=sampler.gates,
                                       rng_after_forward_digest=rng_digest())
                    pair_digest = hashlib.sha256(json.dumps(pair_record, sort_keys=True).encode()).hexdigest()
                    if reference_trace is not None and pair_digest != reference_trace[step]:
                        raise RuntimeError(f"Paired images/noise/padding/time draws/RNG differ at step {step}")
                    trace.append(pair_digest)
                    input_trace.append(digest)
                    t_counts.update(sampler.timesteps)
                    total = sum(losses.values())
                    if not torch.isfinite(total):
                        raise FloatingPointError(f"Non-finite loss at step {step}")
                    total.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(params, cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE,
                                                               error_if_nonfinite=True)
                    optimizer.step()
                    torch.cuda.synchronize()
                    row = dict(step=step, lr=optimizer.param_groups[0]["lr"], loss_total=float(total.detach()),
                               grad_norm=float(grad_norm), step_s=time.perf_counter()-step_start,
                               step_seed=image_seed(train_seed, step), pair_digest=pair_digest,
                               timesteps=sampler.timesteps, **pair_record,
                               **{k: float(v.detach()) for k, v in losses.items()})
                    log.write(json.dumps(row) + "\n")
                    log.flush()
                    if step % 20 == 0 or step == args.steps-1:
                        print(f"[t1-train] seed={train_seed} arm={arm} step={step+1}/{args.steps} "
                              f"loss={row['loss_total']:.4f} tmax={t_counts[999]/sum(t_counts.values()):.3f} "
                              f"step_s={row['step_s']:.3f}", flush=True)
                    scheduler.step()
                    storage.step()
            train_s = time.perf_counter() - tick
            peak_train_gb = torch.cuda.max_memory_allocated() / 1024**3
            reference_trace = trace
            optimizer.zero_grad(set_to_none=True)
            del optimizer, scheduler, loader, batch, losses, total
            student.train_time_sampler = None
            student.train_diffusion_observer = None
            torch.cuda.empty_cache()
            ckpt = arm_dir / "model_final.pth"
            torch.save({"model": student.state_dict(), "iteration": args.steps-1,
                        "train_seed": train_seed, "arm": arm, "config": arm_cfg.dump()}, ckpt)
            training_info = dict(train_wall_s=train_s, peak_train_allocated_gb=peak_train_gb,
                                  initial_state_sha256=initial_digest, discarded_stage_tensors=len(dropped),
                                  checkpoint_sha256=sha256(ckpt), parameters=sum(p.numel() for p in student.parameters()),
                                  batch_trace_sha256=hashlib.sha256("".join(input_trace).encode()).hexdigest(),
                                  pairing_trace_sha256=hashlib.sha256("".join(trace).encode()).hexdigest(),
                                  timestep_histogram=dict(sorted(t_counts.items())),
                                  tmax_fraction=t_counts[999]/sum(t_counts.values()),
                                  timestep_mean=sum(t*n for t,n in t_counts.items())/sum(t_counts.values()),
                                  image_passes=sum(t_counts.values()))
            (arm_dir / "training_summary.json").write_text(json.dumps(training_info, indent=2) + "\n")
            metrics = evaluate(student, arm_cfg, val_records, args, arm_dir)
            for metric in metrics:
                experiment_rows.append(dict(train_seed=train_seed, arm=arm, steps=args.steps,
                                             full_dataset=len(val_records)==len(full_val), train_wall_s=train_s,
                                             **metric))
            with (out / "metrics.csv").open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(experiment_rows[0]))
                writer.writeheader()
                writer.writerows(experiment_rows)
            del student, params
            torch.cuda.empty_cache()
    (out / "COMPLETE.json").write_text(json.dumps({"runs":len(args.train_seeds)*2,
                                                  "evaluation_rows":len(experiment_rows)}) + "\n")


if __name__ == "__main__":
    main()
