# -*- coding: utf-8 -*-
"""
推理延迟 / 显存基准 (M0 -> M2)。

按 FEP sec.2.3 的定义测量：
    latency_ms  : 单图 batch=1，warmup 20 张 + 计时 200 张取**中位数**
    peak_mem_GB : torch.cuda.max_memory_allocated()
    NFE         : 每图的 head 前向次数（backbone 只算一次，见 detector.py:313）

用法：
    python scripts/bench_latency.py --config-file configs/lab/sdd.res50.yaml \
        --weights output/lab/sdd.res50/model_final.pth --steps 1 2 3 4 6 8 10 \
        --num-warmup 20 --num-measure 200
"""

import argparse
import csv
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader
from detectron2.engine import DefaultPredictor  # noqa: F401  (kept for API parity)
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
from detectron2.utils.logger import setup_logger

from diffusiondet import add_diffusiondet_config, DiffusionDetDatasetMapper
from diffusiondet.data_register import register_all
from diffusiondet.util.model_ema import add_model_ema_configs


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config-file", default="configs/lab/sdd.res50.yaml")
    p.add_argument("--weights", default="output/lab/sdd.res50/model_final.pth")
    p.add_argument("--steps", type=int, nargs="+", default=[1, 2, 3, 4, 6, 8, 10])
    p.add_argument("--num-warmup", type=int, default=20)
    p.add_argument("--num-measure", type=int, default=200)
    p.add_argument("--out-csv", default="results/raw/latency.csv")
    p.add_argument("--tag", default="sdd.baseline")
    return p.parse_args()


def main():
    args = parse_args()
    setup_logger()

    from detectron2.config import CfgNode  # noqa: F401

    base_cfg = get_cfg()
    add_diffusiondet_config(base_cfg)
    add_model_ema_configs(base_cfg)
    base_cfg.merge_from_file(args.config_file)
    register_all(cfg=base_cfg)

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    new_file = not os.path.isfile(args.out_csv)
    fout = open(args.out_csv, "a", newline="", encoding="utf-8")
    writer = csv.writer(fout)
    if new_file:
        writer.writerow(["tag", "config", "weights", "sample_step", "nfe",
                         "latency_ms_median", "latency_ms_mean",
                         "peak_mem_gb", "num_warmup", "num_measure"])

    for steps in args.steps:
        cfg = base_cfg.clone()
        cfg.defrost()
        cfg.MODEL.DiffusionDet.SAMPLE_STEP = steps
        cfg.MODEL.DEVICE = "cuda"
        cfg.freeze()

        model = build_model(cfg)
        DetectionCheckpointer(model).load(args.weights)
        model.eval().to("cuda")

        loader = build_detection_test_loader(
            cfg, cfg.DATASETS.TEST[0], mapper=DiffusionDetDatasetMapper(cfg, is_train=False)
        )

        latencies = []
        torch.cuda.reset_peak_memory_stats()
        total = args.num_warmup + args.num_measure
        with torch.no_grad():
            for i, batch in enumerate(loader):
                if i >= total:
                    break
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                model(batch)
                torch.cuda.synchronize()
                dt = (time.perf_counter() - t0) * 1000.0
                if i >= args.num_warmup:
                    latencies.append(dt)

        peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
        med = statistics.median(latencies) if latencies else float("nan")
        mean = statistics.mean(latencies) if latencies else float("nan")
        writer.writerow([args.tag, os.path.basename(args.config_file), os.path.basename(args.weights),
                         steps, steps, round(med, 2), round(mean, 2), round(peak, 3),
                         args.num_warmup, len(latencies)])
        fout.flush()
        print("steps={:<3} n={:<4} median={:7.2f} ms  mean={:7.2f} ms  peak_mem={:.2f} GB".format(
            steps, len(latencies), med, mean, peak))

        del model
        torch.cuda.empty_cache()

    fout.close()
    print("[bench_latency] -> {}".format(args.out_csv))


if __name__ == "__main__":
    main()
