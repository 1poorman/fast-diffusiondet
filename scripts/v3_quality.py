"""Q1: final-head quality focal and equal-budget H6, paired to audited T1 controls."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.v3_distill import batch_digest, evaluate, state_digest
from scripts.v3_head_sweep import image_seed, sha256
from scripts.v3_summarize_distill import read_rows, write_rows
from scripts.v3_summarize_time_alignment import audit_log
from scripts.v3_time_alignment import DiffusionSourceAudit, rng_digest


ARMS = ("h4_focal", "h4_quality", "h6_focal")


def prefix_digest(model):
    kept = {}
    for name, value in model.state_dict().items():
        match = re.match(r"head\.head_series\.(\d+)\.", name)
        if not match or int(match[1]) < 4:
            kept[name] = value
    return state_digest(SimpleNamespace(state_dict=lambda: kept))


def lr_factor(step):
    # The smoke uses the first steps of this same schedule, not a shorter schedule.
    return min(1.0, 0.1 + 0.9 * step / 20) * (0.1 if step >= 480 else 1.0)


def check_config(current, reference):
    """Allow only historical metadata correction and newly disabled config options."""
    import copy
    a, b = copy.deepcopy(current), copy.deepcopy(reference)
    for cfg in (a, b):
        cfg.pop("OUTPUT_DIR")
        cfg["SOLVER"]["WARMUP_FACTOR"] = 0.1  # T1 runner used 0.1, though YAML said 0.01.
        dd = cfg["MODEL"]["DiffusionDet"]
        if dd.pop("FINAL_CLASS_LOSS", "focal") != "focal" or dd.pop("INFERENCE_TOPK", 0) != 0:
            raise ValueError("Reference comparison must use original focal and top-k")
    if a != b:
        raise ValueError("Q1 baseline config differs from T1")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-file", default="configs/lab/v3/sdd.res50.student4.yaml")
    parser.add_argument("--weights", default="output/lab/sdd.res50.bs8/model_final.pth")
    parser.add_argument("--reference-root", type=Path, default=Path("runs/v3/time_alignment_20261011"))
    parser.add_argument("--dataset-root", default="datasets")
    parser.add_argument("--gpu", default="1")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--train-seeds", type=int, nargs="+", default=[10, 11, 12])
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--eval-limit", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.steps <= 600 or args.eval_limit < 0 or args.eval_limit == 1:
        parser.error("steps must be 1..600; eval-limit is zero or >=2")
    if any(len(s) != len(set(s)) for s in (args.train_seeds, args.eval_seeds)):
        parser.error("Seeds must be unique")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    import numpy as np
    import torch
    import yaml
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
    args.out_dir.mkdir(parents=True, exist_ok=False)
    setup_logger(output=str(args.out_dir))
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
    cfg.SOLVER.MAX_ITER, cfg.SOLVER.STEPS = 600, (480,)
    cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_FACTOR = 20, 0.1
    cfg.MODEL.DiffusionDet.TRAIN_T_DIST = "uniform"
    cfg.MODEL.DiffusionDet.TRAIN_T_HIGH_PROB = 0.5
    cfg.OUTPUT_DIR = str(args.out_dir)
    cfg.freeze()
    dd = cfg.MODEL.DiffusionDet
    required = dict(NUM_HEADS=4, EVAL_NUM_HEADS=0, NUM_PROPOSALS=300, INFERENCE_TOPK=0,
                    SAMPLE_STEP=1, FORMULATION="vp", SOLVER="ddim", BOX_RENEWAL=False,
                    USE_ENSEMBLE=False, ATTENTION_NEED_WEIGHTS=False, FINAL_CLASS_LOSS="focal",
                    USE_FOCAL=True, USE_FED_LOSS=False, ALPHA=0.25, GAMMA=2.0)
    if any(getattr(dd, k) != v for k, v in required.items()) or cfg.SOLVER.AMP.ENABLED or cfg.MODEL_EMA.ENABLED:
        raise ValueError("Q1 requires the fixed H4/P300/VP/FP32/uniform protocol")
    reference_manifest = json.loads((args.reference_root / "manifest.json").read_text())
    if (not (args.reference_root / "COMPLETE.json").is_file()
            or reference_manifest["args"]["steps"] != 600
            or sha256(args.weights) != reference_manifest["checkpoint_sha256"]):
        raise ValueError("T1 reference or initialization checkpoint invalid")
    for path, digest in reference_manifest["source_sha256"].items():
        if sha256(args.reference_root / "source_snapshot" / path) != digest:
            raise ValueError(f"T1 source snapshot hash mismatch: {path}")
    register_all(cfg=cfg)
    train_records = sorted(DatasetCatalog.get("sdd_train"), key=lambda r: r["image_id"])
    full_val = sorted(DatasetCatalog.get("sdd_val"), key=lambda r: r["image_id"])
    val_records = full_val[:args.eval_limit] if args.eval_limit else full_val
    if [r["image_id"] for r in full_val] != reference_manifest["eval_image_ids"]:
        raise ValueError("Validation images differ from T1")
    for split, digest in reference_manifest["annotation_sha256"].items():
        if sha256(Path(args.dataset_root) / f"Strawberry/annotations/instances_{split}2017.json") != digest:
            raise ValueError("Data annotations differ from T1")
    (args.out_dir / "config.yaml").write_text(cfg.dump())
    sources = [Path(__file__).relative_to(Path.cwd()), Path(args.config_file),
               Path("configs/lab/sdd.res50.bs8.yaml"), Path("configs/lab/sdd.res50.yaml"),
               Path("configs/diffdet.coco.res50.yaml"),
               *[Path("diffusiondet") / f for f in ("quality.py", "loss.py", "config.py", "head.py",
                   "detector.py", "time_sampling.py", "distillation.py", "dataset_mapper.py", "data_register.py")],
               *[Path("scripts") / f for f in ("v3_distill.py", "v3_time_alignment.py", "v3_head_sweep.py",
                   "v3_summarize_distill.py", "v3_summarize_time_alignment.py")],
               Path("docs/v3/QUALITY_SUPERVISION.md")]
    # Include any config parents that differ from the standard chain above.
    config_path = Path(args.config_file)
    while True:
        parent = yaml.safe_load(config_path.read_text()).get("_BASE_")
        if parent is None:
            break
        config_path = (config_path.parent / parent).resolve().relative_to(Path.cwd())
        if config_path not in sources:
            sources.append(config_path)
    sources = list(dict.fromkeys(sources))
    for source in sources:
        target = args.out_dir / "source_snapshot" / source
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    references, reference_hashes = {}, {}
    for seed in args.train_seeds:
        arm = args.reference_root / f"seed{seed}_uniform"
        info = json.loads((arm / "training_summary.json").read_text())
        logs = [json.loads(line) for line in (arm / "training.jsonl").read_text().splitlines()]
        computed = audit_log(logs, "uniform", 0.5, cfg.SOLVER.IMS_PER_BATCH)
        if len(logs) != 600 or any(info[k] != v for k, v in computed.items()):
            raise ValueError("Invalid T1 training audit")
        check_config(yaml.safe_load(cfg.dump()), yaml.safe_load((arm / "config.yaml").read_text()))
        checkpoint = arm / "model_final.pth"
        if sha256(checkpoint) != info["checkpoint_sha256"]:
            raise ValueError("T1 control checkpoint hash mismatch")
        references[seed] = (info, logs, checkpoint)
        for name in ("training_summary.json", "training.jsonl", "config.yaml"):
            target = args.out_dir / "reference_snapshot" / f"seed{seed}_uniform" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(arm / name, target)
            reference_hashes[str(target.relative_to(args.out_dir))] = sha256(target)
    manifest = dict(args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    source_sha256={str(p): sha256(p) for p in sources},
                    reference_source_sha256=reference_manifest["source_sha256"], reference_snapshot_sha256=reference_hashes,
                    initialization_sha256=sha256(args.weights), reference_manifest_sha256=sha256(args.reference_root / "manifest.json"),
                    reference_checkpoint_sha256={str(s): x[0]["checkpoint_sha256"] for s, x in references.items()},
                    annotation_sha256=reference_manifest["annotation_sha256"],
                    full_val_images=len(full_val), image_ids=[r["image_id"] for r in val_records],
                    train_images=len(train_records),
                    arm_order={str(s): list(ARMS if s % 2 == 0 else reversed(ARMS)) for s in args.train_seeds},
                    torch=torch.__version__, detectron2=detectron2.__version__, cuda=torch.version.cuda,
                    gpu=torch.cuda.get_device_name(), git_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                    protocol="Q1 final-only quality focal; reused T1 H4 control; newly trained H4 quality/H6 focal; identical 600-step schedule and audited per-step RNG/inputs; FP32, TF32 off, batch1 eval")
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    source_state = torch.load(args.weights, map_location="cpu")["model"]
    rows, replay_rows = [], []

    def make_loader(arm_cfg, seed):
        seed_all(seed)
        return iter(build_detection_train_loader(dataset=train_records, mapper=DiffusionDetDatasetMapper(arm_cfg, is_train=True),
                    sampler=TrainingSampler(len(train_records), seed=seed), total_batch_size=cfg.SOLVER.IMS_PER_BATCH,
                    aspect_ratio_grouping=True, num_workers=0))

    def audited_forward(model, loader, sampler, audit, seed, step, reference):
        seed_all(image_seed(seed, step))
        sampler.reset(image_seed(seed, f"tgate:{step}"))
        audit.reset()
        batch = next(loader)
        digest = batch_digest(batch)
        losses = model(batch)
        record = dict(input_digest=digest, source_digest=audit.digest.hexdigest(), base_timesteps=sampler.base_timesteps,
                      gates=sampler.gates, rng_after_forward_digest=rng_digest())
        pair = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
        if (pair != reference["pair_digest"] or sampler.timesteps != reference["timesteps"]
                or audit.count != len(batch) or len(sampler.timesteps) != len(batch)):
            raise ValueError(f"Q1/T1 input, timestep or RNG pairing failed: seed={seed}, step={step}")
        return losses, dict(pair_digest=pair, timesteps=sampler.timesteps, **record)

    for seed in args.train_seeds:
        ref_info, ref_logs, ref_checkpoint = references[seed]
        # Real GPU replay of the default path before using the historical control.
        seed_all(seed)
        replay = build_model(cfg).train()
        load_truncated_student(replay, source_state)
        if state_digest(replay) != ref_info["initial_state_sha256"]:
            raise ValueError("T1/Q1 initial H4 state differs")
        sampler, audit = PairedTimeSampler(), DiffusionSourceAudit()
        replay.train_time_sampler, replay.train_diffusion_observer = sampler, audit
        loader = make_loader(cfg, seed)
        with torch.no_grad(), EventStorage():
            losses, trace = audited_forward(replay, loader, sampler, audit, seed, 0, ref_logs[0])
        differences = {k: float(v) - ref_logs[0][k] for k, v in losses.items()}
        if any(not math.isclose(float(losses[k]), ref_logs[0][k], rel_tol=1e-6, abs_tol=1e-6) for k in losses):
            raise ValueError(f"Default focal replay differs from T1: {differences}")
        replay_rows.append(dict(train_seed=seed, initial_state_sha256=state_digest(replay),
                                loss_differences=differences, **trace))
        (args.out_dir / "control_replay.json").write_text(json.dumps(replay_rows, indent=2) + "\n")
        del replay, loader, losses
        torch.cuda.empty_cache()
        for arm in manifest["arm_order"][str(seed)]:
            arm_dir = args.out_dir / f"seed{seed}_{arm}"
            arm_dir.mkdir()
            arm_cfg = cfg.clone()
            arm_cfg.defrost()
            arm_cfg.MODEL.DiffusionDet.NUM_HEADS = 6 if arm == "h6_focal" else 4
            arm_cfg.MODEL.DiffusionDet.FINAL_CLASS_LOSS = "quality_focal" if arm == "h4_quality" else "focal"
            arm_cfg.OUTPUT_DIR = str(arm_dir)
            arm_cfg.freeze()
            (arm_dir / "config.yaml").write_text(arm_cfg.dump())
            seed_all(seed)
            model = build_model(arm_cfg)
            if arm == "h4_focal":
                checkpoint = torch.load(ref_checkpoint, map_location="cpu")
                if checkpoint["arm"] != "uniform" or checkpoint["train_seed"] != seed or checkpoint["iteration"] != 599:
                    raise ValueError("Wrong reference checkpoint metadata")
                model.load_state_dict(checkpoint["model"], strict=True)
                del checkpoint
                info = dict(ref_info, trained_this_run=False, checkpoint_path=str(ref_checkpoint),
                            executed_steps=600, heads=4, final_class_loss="focal")
            else:
                dropped = load_truncated_student(model, source_state)
                initial = state_digest(model)
                if prefix_digest(model) != ref_info["initial_state_sha256"]:
                    raise ValueError("H4 prefix initialization differs from T1")
                sampler, audit = PairedTimeSampler(), DiffusionSourceAudit()
                model.train_time_sampler, model.train_diffusion_observer = sampler, audit
                model.train()
                loader = make_loader(arm_cfg, seed)
                params = [p for p in model.parameters() if p.requires_grad]
                optimizer = torch.optim.AdamW(params, lr=cfg.SOLVER.BASE_LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
                scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
                logs = []
                torch.cuda.reset_peak_memory_stats()
                tick = time.perf_counter()
                with (arm_dir / "training.jsonl").open("w") as log, EventStorage() as storage:
                    for step in range(args.steps):
                        step_start = time.perf_counter()
                        optimizer.zero_grad(set_to_none=True)
                        losses, trace = audited_forward(model, loader, sampler, audit, seed, step, ref_logs[step])
                        total = sum(losses.values())
                        if not torch.isfinite(total):
                            raise FloatingPointError("Non-finite training loss")
                        total.backward()
                        grad = torch.nn.utils.clip_grad_norm_(params, cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE, error_if_nonfinite=True)
                        optimizer.step()
                        torch.cuda.synchronize()
                        lr = optimizer.param_groups[0]["lr"]
                        if lr != ref_logs[step]["lr"]:
                            raise ValueError("Learning-rate schedule differs from T1")
                        row = dict(step=step, step_seed=image_seed(seed, step), lr=lr, loss_total=float(total.detach()),
                                   grad_norm=float(grad), step_s=time.perf_counter()-step_start, **trace,
                                   **{k: float(v.detach()) for k, v in losses.items()},
                                   **{k: float(v) for k, v in model.criterion.last_quality_stats.items()})
                        logs.append(row)
                        log.write(json.dumps(row) + "\n")
                        log.flush()
                        if step % 20 == 0 or step == args.steps - 1:
                            print(f"[q1-train] seed={seed} arm={arm} step={step+1}/{args.steps} loss={row['loss_total']:.4f} step_s={row['step_s']:.3f}", flush=True)
                        scheduler.step()
                        storage.step()
                train_s = time.perf_counter() - tick
                peak = torch.cuda.max_memory_allocated() / 1024**3
                optimizer.zero_grad(set_to_none=True)
                del optimizer, scheduler, loader, losses, total, params
                model.train_time_sampler = model.train_diffusion_observer = None
                checkpoint = arm_dir / "model_final.pth"
                torch.save(dict(model=model.state_dict(), iteration=args.steps-1, train_seed=seed, arm=arm, config=arm_cfg.dump()), checkpoint)
                info = dict(train_wall_s=train_s, peak_train_allocated_gb=peak, initial_state_sha256=initial,
                            initial_h4_prefix_sha256=ref_info["initial_state_sha256"], discarded_stage_tensors=len(dropped),
                            checkpoint_sha256=sha256(checkpoint), checkpoint_path=str(checkpoint),
                            parameters=sum(p.numel() for p in model.parameters()), trained_this_run=True,
                            heads=arm_cfg.MODEL.DiffusionDet.NUM_HEADS, final_class_loss=arm_cfg.MODEL.DiffusionDet.FINAL_CLASS_LOSS,
                            executed_steps=args.steps, **audit_log(logs, "uniform", 0.5, cfg.SOLVER.IMS_PER_BATCH))
            (arm_dir / "training_summary.json").write_text(json.dumps(info, indent=2) + "\n")
            torch.cuda.empty_cache()
            metrics = evaluate(model, arm_cfg, val_records, args, arm_dir)
            for metric in metrics:
                rows.append(dict(train_seed=seed, arm=arm, steps=info["executed_steps"],
                                 full_dataset=len(val_records) == len(full_val), **metric))
            write_rows(args.out_dir / "metrics.csv", rows)
            del model
            torch.cuda.empty_cache()
    (args.out_dir / "COMPLETE.json").write_text(json.dumps(dict(evaluations=len(rows), new_training_runs=2*len(args.train_seeds),
                                                              reused_controls=len(args.train_seeds))) + "\n")


if __name__ == "__main__":
    main()
