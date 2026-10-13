"""Paired pilot: a real four-stage student with/without a frozen six-stage teacher.

Uses equal image passes, augmentation/noise seeds and initialization. All outputs
stay in a new run directory. This is a short-budget screen, not a full G2 claim.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.v3_head_sweep import image_seed, sha256


def state_digest(model):
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def batch_digest(batch):
    h = hashlib.sha256()
    for record in batch:
        h.update(str(record["image_id"]).encode())
        for tensor in (record["image"], record["instances"].gt_boxes.tensor, record["instances"].gt_classes):
            h.update(str(tuple(tensor.shape)).encode())
            h.update(tensor.contiguous().numpy().tobytes())
    return h.hexdigest()


def evaluate(model, cfg, records, args, arm_dir):
    import numpy as np
    import torch
    from detectron2.data import DatasetMapper, build_detection_test_loader
    from detectron2.evaluation import COCOEvaluator
    # Match the frozen-checkpoint G1 evaluation arithmetic/backend protocol.
    torch.backends.cudnn.deterministic = False
    model.eval()
    loader = build_detection_test_loader(dataset=records, mapper=DatasetMapper(cfg, is_train=False),
                                         batch_size=1, num_workers=0)
    results = []
    for seed in args.eval_seeds:
        evaluator = COCOEvaluator("sdd_val", tasks=("bbox",), distributed=False,
                                  output_dir=str(arm_dir / f"eval_seed{seed}"))
        evaluator.reset()
        latencies = []
        warmup = min(20, max(1, len(records) // 4))
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            for index, batch in enumerate(loader):
                torch.manual_seed(image_seed(seed, batch[0]["image_id"]))
                torch.cuda.synchronize()
                tick = time.perf_counter()
                predictions = model(batch)
                torch.cuda.synchronize()
                if warmup <= index < warmup + 200:
                    latencies.append(1000 * (time.perf_counter() - tick))
                evaluator.process(batch, predictions)
        metrics = evaluator.evaluate(img_ids=[r["image_id"] for r in records])["bbox"]
        row = dict(eval_seed=seed, images=len(records), **metrics,
                   latency_p50_ms=float(np.median(latencies)), latency_p95_ms=float(np.percentile(latencies, 95)),
                   measured_images=len(latencies), peak_allocated_gb=torch.cuda.max_memory_allocated()/1024**3)
        results.append(row)
        print("[v3-eval] " + json.dumps(row), flush=True)
    (arm_dir / "evaluation.json").write_text(json.dumps(results, indent=2) + "\n")
    torch.backends.cudnn.deterministic = True
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config-file", default="configs/lab/v3/sdd.res50.student4.yaml")
    p.add_argument("--weights", default="output/lab/sdd.res50.bs8/model_final.pth")
    p.add_argument("--dataset-root", default="datasets")
    p.add_argument("--gpu", default="1")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--train-seeds", type=int, nargs="+", default=[10])
    p.add_argument("--eval-seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--arms", nargs="+", choices=["control", "kd"], default=["control", "kd"])
    p.add_argument("--eval-limit", type=int, default=0)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()
    if args.steps < 1 or args.eval_limit < 0 or args.eval_limit == 1:
        p.error("steps must be positive; eval-limit must be zero or at least two")
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
    cfg.OUTPUT_DIR = str(out)
    cfg.freeze()
    if cfg.MODEL.DiffusionDet.NUM_HEADS != 4 or cfg.MODEL.DiffusionDet.FORMULATION != "vp":
        raise ValueError("Pilot protocol requires a four-stage VP student")
    if cfg.SOLVER.AMP.ENABLED or cfg.MODEL_EMA.ENABLED:
        raise ValueError("Pilot uses FP32 and raw student weights only")
    register_all(cfg=cfg)
    train_records = sorted(DatasetCatalog.get("sdd_train"), key=lambda r: r["image_id"])
    full_val = sorted(DatasetCatalog.get("sdd_val"), key=lambda r: r["image_id"])
    val_records = full_val[:args.eval_limit] if args.eval_limit else full_val
    state = torch.load(args.weights, map_location="cpu")["model"]
    (out / "config.yaml").write_text(cfg.dump())
    sources = [Path(__file__), Path("diffusiondet/distillation.py"), Path("diffusiondet/detector.py"),
               Path("diffusiondet/loss.py"), Path("diffusiondet/head.py"), Path("diffusiondet/config.py"),
               Path("diffusiondet/dataset_mapper.py"), Path("scripts/v3_head_sweep.py")]
    manifest = dict(args=vars(args), source_sha256={str(p): sha256(p) for p in sources},
                    checkpoint_sha256=sha256(args.weights), checkpoint=str(Path(args.weights).resolve()),
                    git_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                    torch=torch.__version__, detectron2=detectron2.__version__, cuda=torch.version.cuda,
                    gpu=torch.cuda.get_device_name(), full_val_images=len(full_val),
                    train_images=len(train_records), eval_image_ids=[r["image_id"] for r in val_records],
                    protocol="FP32; TF32 off; cuDNN deterministic during training, default during eval; fresh AdamW; uniform training t; only final-stage KD",
                    training_seed_protocol="image_seed(train_seed, iteration), reset Python/NumPy/torch before each batch",
                    annotation_sha256={split: sha256(Path(args.dataset_root) / "Strawberry/annotations" / f"instances_{split}2017.json") for split in ("train", "val")})
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    experiment_rows = []
    for train_seed in args.train_seeds:
        reference_trace = None
        reference_init = None
        for arm in args.arms:
            arm_dir = out / f"seed{train_seed}_{arm}"
            arm_dir.mkdir()
            seed_all(train_seed)
            student = build_model(cfg)
            dropped = load_truncated_student(student, state)
            initial_digest = state_digest(student)
            if reference_init is None:
                reference_init = initial_digest
            elif initial_digest != reference_init:
                raise RuntimeError("Paired student initialization differs")
            teacher = None
            teacher_digest = None
            if arm == "kd":
                teacher_cfg = cfg.clone()
                teacher_cfg.defrost()
                teacher_cfg.MODEL.DiffusionDet.NUM_HEADS = 6
                teacher_cfg.MODEL.DiffusionDet.EVAL_NUM_HEADS = 0
                teacher_cfg.freeze()
                teacher = build_model(teacher_cfg).eval().requires_grad_(False)
                teacher.load_state_dict(state, strict=True)
                teacher_digest = state_digest(teacher)
            seed_all(train_seed)
            loader = iter(build_detection_train_loader(
                dataset=train_records, mapper=DiffusionDetDatasetMapper(cfg, is_train=True),
                sampler=TrainingSampler(len(train_records), seed=train_seed),
                total_batch_size=cfg.SOLVER.IMS_PER_BATCH, aspect_ratio_grouping=True, num_workers=0))
            params = [p for p in student.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(params, lr=cfg.SOLVER.BASE_LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY)

            def lr_factor(step):
                warm = min(1.0, 0.1 + 0.9 * step / max(1, cfg.SOLVER.WARMUP_ITERS))
                return warm * (0.1 if step >= cfg.SOLVER.STEPS[0] else 1.0)

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
            trace = []
            student.train()
            torch.cuda.reset_peak_memory_stats()
            tick = time.perf_counter()
            with (arm_dir / "training.jsonl").open("w") as log, EventStorage() as storage:
                for step in range(args.steps):
                    seed_all(image_seed(train_seed, step))
                    step_start = time.perf_counter()
                    batch = next(loader)
                    digest = batch_digest(batch)
                    trace.append(digest)
                    if reference_trace is not None and digest != reference_trace[step]:
                        raise RuntimeError(f"Training inputs/augmentations differ at step {step}")
                    optimizer.zero_grad(set_to_none=True)
                    losses = student(batch, teacher=teacher)
                    total = sum(losses.values())
                    if not torch.isfinite(total):
                        raise FloatingPointError(f"Non-finite loss at step {step}: {losses}")
                    total.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(params, cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE,
                                                               error_if_nonfinite=True)
                    optimizer.step()
                    torch.cuda.synchronize()
                    row = dict(step=step, lr=optimizer.param_groups[0]["lr"], loss_total=float(total.detach()),
                               grad_norm=float(grad_norm), step_s=time.perf_counter()-step_start,
                               input_digest=digest, step_seed=image_seed(train_seed, step),
                               **{k: float(v.detach()) for k, v in losses.items()}, **student.last_distillation_stats)
                    log.write(json.dumps(row) + "\n")
                    log.flush()
                    if step % 20 == 0 or step == args.steps-1:
                        print(f"[v3-train] seed={train_seed} arm={arm} step={step+1}/{args.steps} "
                              f"loss={row['loss_total']:.4f} kd_pairs={row.get('kd_pairs', 0)} "
                              f"step_s={row['step_s']:.3f}", flush=True)
                    scheduler.step()
                    storage.step()
            train_s = time.perf_counter() - tick
            peak_train_gb = torch.cuda.max_memory_allocated() / 1024**3
            if teacher is not None:
                if any(p.grad is not None for p in teacher.parameters()) or state_digest(teacher) != teacher_digest:
                    raise RuntimeError("Frozen teacher changed or received gradients")
            reference_trace = trace
            teacher = None
            optimizer.zero_grad(set_to_none=True)
            del optimizer, scheduler, loader, batch, losses, total
            torch.cuda.empty_cache()
            ckpt = arm_dir / "model_final.pth"
            torch.save({"model": student.state_dict(), "iteration": args.steps-1,
                        "train_seed": train_seed, "arm": arm, "config": cfg.dump()}, ckpt)
            training_info = dict(train_wall_s=train_s, peak_train_allocated_gb=peak_train_gb,
                                  initial_state_sha256=initial_digest, discarded_stage_tensors=len(dropped),
                                  teacher_unchanged=True if arm == "kd" else None,
                                  checkpoint_sha256=sha256(ckpt), parameters=sum(p.numel() for p in student.parameters()),
                                  batch_trace_sha256=hashlib.sha256("".join(trace).encode()).hexdigest(),
                                  image_passes=args.steps * cfg.SOLVER.IMS_PER_BATCH)
            (arm_dir / "training_summary.json").write_text(json.dumps(training_info, indent=2) + "\n")
            metrics = evaluate(student, cfg, val_records, args, arm_dir)
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
    (out / "COMPLETE.json").write_text(json.dumps({"runs":len(args.train_seeds)*len(args.arms),
                                                  "evaluation_rows":len(experiment_rows)}) + "\n")


if __name__ == "__main__":
    main()
