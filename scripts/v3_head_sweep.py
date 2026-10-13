"""Paired, batch=1 AP/latency screening of checkpoint-compatible head exits.

Example: python scripts/v3_head_sweep.py --gpu 1 --out-dir runs/v3/screen
Every image gets noise seeded by (eval seed, image_id), independently of arm order.
Timing includes H2D/preprocessing/backbone/head/NMS/postprocessing, excludes disk
loading, resize and evaluator. No inference autocast; FP32/TF32 flags are recorded.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def image_seed(seed, image_id):
    key = f"fastdd-v3:{seed}:{image_id}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "little")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config-file", default="configs/lab/sdd.res50.bs8.yaml")
    p.add_argument("--weights", default="output/lab/sdd.res50.bs8/model_final.pth")
    p.add_argument("--dataset-root", default="datasets")
    p.add_argument("--dataset", default="sdd_val")
    p.add_argument("--gpu", default="1")
    p.add_argument("--arms", nargs="+", default=["6:legacy", "6:no_weights", "5:no_weights", "4:no_weights", "3:no_weights"])
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--limit", type=int, default=0, help="0 = full validation; subsets never count as final AP")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--measure", type=int, default=200)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()
    if args.limit < 0 or args.warmup < 0 or args.measure <= 0:
        p.error("limit/warmup must be nonnegative; measure must be positive")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    import numpy as np
    import torch
    import detectron2
    from detectron2.config import get_cfg
    from detectron2.data import DatasetCatalog, DatasetMapper, MetadataCatalog, build_detection_test_loader
    from detectron2.evaluation import COCOEvaluator
    from detectron2.modeling import build_model
    from detectron2.utils.logger import setup_logger
    from diffusiondet import add_diffusiondet_config
    from diffusiondet.data_register import register_all
    from diffusiondet.util.model_ema import add_model_ema_configs

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; do not report CPU timing as GPU latency")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=False)  # Protect prior results from overwrite.
    setup_logger(output=str(out))
    torch.set_num_threads(args.threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    cfg = get_cfg()
    add_diffusiondet_config(cfg)
    add_model_ema_configs(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.DATASET_ROOT = str(Path(args.dataset_root).resolve())
    cfg.MODEL.WEIGHTS = args.weights
    cfg.MODEL.DEVICE = "cuda"
    cfg.MODEL.DiffusionDet.SOLVER = "ddim"
    cfg.MODEL.DiffusionDet.SAMPLE_STEP = 1
    cfg.MODEL.DiffusionDet.BOX_RENEWAL = False
    cfg.MODEL.DiffusionDet.USE_ENSEMBLE = False
    cfg.OUTPUT_DIR = str(out)
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.freeze()
    register_all(cfg=cfg)
    records = sorted(DatasetCatalog.get(args.dataset), key=lambda x: x["image_id"])
    full_count = len(records)
    if args.limit:
        records = records[:args.limit]
    if len(records) <= args.warmup:
        raise ValueError("Need more images than warmup for valid timing")
    model = build_model(cfg).eval()
    checkpoint = torch.load(args.weights, map_location="cpu")
    # Evaluation must never silently skip the trained classifier or any stage.
    model.load_state_dict(checkpoint["model"], strict=True)
    del checkpoint
    arms = []
    for arm in args.arms:
        depth, attention = arm.split(":")
        depth = int(depth)
        if not 1 <= depth <= model.head.num_heads or attention not in ("legacy", "no_weights"):
            raise ValueError(f"Invalid arm {arm}")
        arms.append((depth, attention))
    mapper = DatasetMapper(cfg, is_train=False)
    loader = build_detection_test_loader(dataset=records, mapper=mapper, num_workers=0, batch_size=1)
    ids = [r["image_id"] for r in records]
    annotation = MetadataCatalog.get(args.dataset).json_file
    manifest = dict(args=vars(args), git_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                    checkpoint=str(Path(args.weights).resolve()), checkpoint_sha256=sha256(args.weights),
                    annotations_sha256=sha256(annotation), image_ids=ids, full_dataset_count=full_count,
                    torch=torch.__version__, detectron2=detectron2.__version__, python=platform.python_version(),
                    gpu=torch.cuda.get_device_name(), cuda=torch.version.cuda,
                    cudnn_benchmark=False, matmul_tf32=False, cudnn_tf32=False, autocast=False,
                    timing="batch=1 synchronized wall clock; model call only; no disk/resize/evaluator",
                    seed_protocol="sha256(fastdd-v3:eval_seed:image_id) first 4 bytes little-endian",
                    checkpoint_load="strict=True, raw model (no EMA)")
    manifest["source_sha256"] = {str(path): sha256(path) for path in
        [Path(__file__), Path("diffusiondet/head.py"), Path("diffusiondet/config.py"), Path("diffusiondet/detector.py")]}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (out / "config.yaml").write_text(cfg.dump())
    fields = ["seed", "depth", "attention", "nfe", "proposals", "images", "full_dataset", "AP", "AP50", "AP75", "APs", "APm", "APl", "latency_p50_ms", "latency_p95_ms", "latency_mean_ms", "measured_images", "peak_allocated_gb", "wall_s"]
    with open(out / "metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for seed in args.seeds:
            # Reverse order on odd seeds to reduce systematic order effects.
            for depth, attention in (arms if seed % 2 == 0 else list(reversed(arms))):
                arm_dir = out / f"seed{seed}_d{depth}_{attention}"
                arm_dir.mkdir()
                evaluator = COCOEvaluator(args.dataset, tasks=("bbox",), distributed=False, output_dir=str(arm_dir))
                evaluator.reset()
                model.head.eval_num_heads = depth
                for stage in model.head.head_series:
                    stage.attention_need_weights = attention == "legacy"
                latencies = []
                torch.cuda.reset_peak_memory_stats()
                start = time.perf_counter()
                with torch.no_grad():
                    for index, batch in enumerate(loader):
                        torch.manual_seed(image_seed(seed, batch[0]["image_id"]))
                        torch.cuda.synchronize()
                        tick = time.perf_counter()
                        predictions = model(batch)
                        torch.cuda.synchronize()
                        ms = (time.perf_counter() - tick) * 1000
                        if args.warmup <= index < args.warmup + args.measure:
                            latencies.append(ms)
                        evaluator.process(batch, predictions)
                # Explicit subset image IDs prevent unevaluated images counting as misses.
                result = evaluator.evaluate(img_ids=ids)["bbox"]
                (arm_dir / "bbox_metrics.json").write_text(json.dumps(result, indent=2) + "\n")
                row = dict(seed=seed, depth=depth, attention=attention, nfe=1, proposals=model.num_proposals,
                           images=len(records), full_dataset=len(records) == full_count,
                           **{key: result.get(key, float("nan")) for key in ("AP", "AP50", "AP75", "APs", "APm", "APl")},
                           latency_p50_ms=float(np.median(latencies)), latency_p95_ms=float(np.percentile(latencies, 95)),
                           latency_mean_ms=float(np.mean(latencies)), measured_images=len(latencies),
                           peak_allocated_gb=torch.cuda.max_memory_allocated() / 1024**3,
                           wall_s=time.perf_counter() - start)
                writer.writerow(row)
                f.flush()
                print("[v3] " + json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
