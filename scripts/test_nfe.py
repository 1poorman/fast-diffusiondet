# -*- coding: utf-8 -*-
"""
NFE 计数校准 (M1 Gate)。

NFE 定义：每张图的 **head 前向次数**（backbone 只算一次，见 detector.py forward）。
理论值：
    ddim steps=k           -> k
    euler num_steps=n      -> n
    heun  num_steps=n      -> 2n - 1
    dpm_solver_v3 steps=s  -> s

用法：
    python scripts/test_nfe.py --config-file configs/lab/sdd.res50.yaml \
        --weights output/lab/sdd.res50/model_final.pth
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader
from detectron2.modeling import build_model
from detectron2.utils.logger import setup_logger

from diffusiondet import add_diffusiondet_config, DiffusionDetDatasetMapper
from diffusiondet.data_register import register_all
from diffusiondet.util.model_ema import add_model_ema_configs

CASES = [
    ("ddim", 1, 1), ("ddim", 4, 4),
    ("euler", 4, 4),
    ("heun", 2, 3), ("heun", 3, 5),
    ("dpm_solver_v3", 4, 4), ("dpm_solver_v3", 8, 8),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-file", default="configs/lab/sdd.res50.yaml")
    ap.add_argument("--weights", default="output/lab/sdd.res50/model_final.pth")
    ap.add_argument("--num-images", type=int, default=2)
    args = ap.parse_args()
    setup_logger()

    base_cfg = get_cfg()
    add_diffusiondet_config(base_cfg)
    add_model_ema_configs(base_cfg)
    base_cfg.merge_from_file(args.config_file)
    register_all(cfg=base_cfg)

    ok = True
    print("{:<16} {:<8} {:<8} {:<8} {}".format("solver", "param", "expect", "actual", "result"))
    for solver, param, expect in CASES:
        cfg = base_cfg.clone()
        cfg.defrost()
        cfg.MODEL.DEVICE = "cuda"
        cfg.MODEL.DiffusionDet.SOLVER = solver
        cfg.MODEL.DiffusionDet.SAMPLE_STEP = param
        cfg.freeze()
        model = build_model(cfg)
        DetectionCheckpointer(model).load(args.weights)
        model.eval().to("cuda")
        loader = build_detection_test_loader(
            cfg, cfg.DATASETS.TEST[0], mapper=DiffusionDetDatasetMapper(cfg, is_train=False))

        nf = []
        with torch.no_grad():
            for i, batch in enumerate(loader):
                if i >= args.num_images:
                    break
                inputs = batch if isinstance(batch, list) else [batch]
                model(inputs)
                nf.append(model._last_nfe)
        actual = nf[0] if nf else -1
        good = (actual == expect) and all(n == expect for n in nf)
        ok = ok and good
        print("{:<16} {:<8} {:<8} {:<8} {}".format(
            solver, param, expect, actual, "PASS" if good else "FAIL (nfe per image={})".format(nf)))
        del model
        torch.cuda.empty_cache()

    print("[test_nfe] {}".format("PASS" if ok else "FAIL"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
