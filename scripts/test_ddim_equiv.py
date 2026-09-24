# -*- coding: utf-8 -*-
"""
DDIM 逐 bit 等价回归测试 (M1 Gate)。

用法：
    # 1) 在改动 detector.py 之前，固化基线行为
    python scripts/test_ddim_equiv.py --save-reference
    # 2) 重构到 diffusiondet/solvers/ 之后，验证等价
    python scripts/test_ddim_equiv.py --check

判定：同 seed、同 NFE 下，box 坐标 max|Δ| < 1e-5。
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

REF_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "tests", "fixtures", "ddim_reference.pt")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config-file", default="configs/lab/sdd.res50.yaml")
    p.add_argument("--weights", default="output/lab/sdd.res50/model_final.pth")
    p.add_argument("--steps", type=int, nargs="+", default=[1, 4])
    p.add_argument("--num-images", type=int, default=8)
    p.add_argument("--base-seed", type=int, default=42)
    p.add_argument("--save-reference", action="store_true")
    p.add_argument("--check", action="store_true")
    return p.parse_args()


@torch.no_grad()
def collect(model, loader, steps, num_images, base_seed):
    out = {}
    for i, batch in enumerate(loader):
        if i >= num_images:
            break
        for s in steps:
            model.sampling_timesteps = s
            model.is_ddim_sampling = s < model.num_timesteps
            torch.manual_seed(base_seed + i)
            inputs = batch if isinstance(batch, list) else [batch]
            out0 = model(inputs)[0]
            res = out0["instances"] if isinstance(out0, dict) else out0
            out[(i, s)] = (
                res.pred_boxes.tensor.detach().cpu().clone(),
                res.scores.detach().cpu().clone(),
                res.pred_classes.detach().cpu().clone(),
            )
    return out


def main():
    args = parse_args()
    setup_logger()

    base_cfg = get_cfg()
    add_diffusiondet_config(base_cfg)
    add_model_ema_configs(base_cfg)
    base_cfg.merge_from_file(args.config_file)
    register_all(cfg=base_cfg)

    cfg = base_cfg.clone()
    cfg.defrost()
    cfg.MODEL.DEVICE = "cuda"
    cfg.freeze()

    model = build_model(cfg)
    DetectionCheckpointer(model).load(args.weights)
    model.eval().to("cuda")

    loader = build_detection_test_loader(
        cfg, cfg.DATASETS.TEST[0], mapper=DiffusionDetDatasetMapper(cfg, is_train=False)
    )
    cur = collect(model, loader, args.steps, args.num_images, args.base_seed)

    if args.save_reference:
        os.makedirs(os.path.dirname(REF_PATH), exist_ok=True)
        torch.save({"steps": args.steps, "num_images": args.num_images,
                    "base_seed": args.base_seed, "data": cur}, REF_PATH)
        print("[test_ddim_equiv] reference saved -> {}".format(REF_PATH))
        return

    if args.check:
        ref = torch.load(REF_PATH)
        assert ref["base_seed"] == args.base_seed and ref["num_images"] == args.num_images, \
            "reference 参数不一致，请重新 capture"
        worst = 0.0
        worst_key = None
        missing = 0
        for k, v in ref["data"].items():
            if k not in cur:
                missing += 1
                continue
            for a, b in zip(v, cur[k]):
                if a.numel() != b.numel():
                    d = float("inf")
                else:
                    d = (a - b).abs().max().item() if a.numel() else 0.0
                if d > worst:
                    worst, worst_key = d, k
        print("[test_ddim_equiv] compared {} entries, missing={}".format(len(ref["data"]), missing))
        print("[test_ddim_equiv] max|delta| = {:.3e} (worst at {})".format(worst, worst_key))
        if missing == 0 and worst < 1e-5:
            print("[test_ddim_equiv] PASS (max|delta| < 1e-5)")
        else:
            print("[test_ddim_equiv] FAIL")
            sys.exit(1)
        return

    print("please pass --save-reference or --check")


if __name__ == "__main__":
    main()
