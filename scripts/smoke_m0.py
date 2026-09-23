# -*- coding: utf-8 -*-
"""
M0 冒烟测试：不做训练，只验证三件事能否跑通。

1. 数据集注册（含 background 类别剔除与 id 映射）
2. 配置文件能否被 detectron2 正确解析
3. COCO 迁移初始化（D2）能否加载，哪些层因形状不匹配被重新初始化

用法：
    python scripts/smoke_m0.py --config-file configs/lab/sdd.res50.yaml
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config-file", default="configs/lab/sdd.res50.yaml")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--skip-model", action="store_true", help="只验证数据集注册，不建模型")
    return p.parse_args()


def main():
    args = parse_args()

    from detectron2.config import get_cfg
    from detectron2.utils.logger import setup_logger

    from diffusiondet import add_diffusiondet_config
    from diffusiondet.data_register import register_all
    from diffusiondet.util.model_ema import add_model_ema_configs
    from diffusiondet.weights import resolve_pretrain

    setup_logger()

    cfg = get_cfg()
    add_diffusiondet_config(cfg)
    add_model_ema_configs(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.MODEL.DEVICE = args.device
    cfg.freeze()

    print("=" * 78)
    print("[1/3] 数据集注册")
    print("=" * 78)
    registered = register_all(cfg=cfg)
    for name, n_img, n_cls in registered:
        print("      {:<12} images={:<6} classes={}".format(name, n_img, n_cls))

    from detectron2.data import DatasetCatalog, MetadataCatalog
    for name, _, _ in registered:
        d = DatasetCatalog.get(name)
        meta = MetadataCatalog.get(name)
        print("      {:<12} thing_classes={}".format(name, meta.thing_classes))
        assert len(d) > 0, "空数据集: " + name

    print()
    print("=" * 78)
    print("[2/3] 配置解析")
    print("=" * 78)
    print("      NUM_CLASSES   = {}".format(cfg.MODEL.DiffusionDet.NUM_CLASSES))
    print("      NUM_PROPOSALS = {}".format(cfg.MODEL.DiffusionDet.NUM_PROPOSALS))
    print("      IMS_PER_BATCH = {}".format(cfg.SOLVER.IMS_PER_BATCH))
    print("      MAX_ITER      = {}".format(cfg.SOLVER.MAX_ITER))
    print("      AMP.ENABLED   = {}".format(cfg.SOLVER.AMP.ENABLED))
    print("      OUTPUT_DIR    = {}".format(cfg.OUTPUT_DIR))
    print("      TRAIN/TEST    = {} / {}".format(cfg.DATASETS.TRAIN, cfg.DATASETS.TEST))

    print()
    print("=" * 78)
    print("[3/3] COCO 迁移初始化 (D2)")
    print("=" * 78)
    path = resolve_pretrain(cfg)
    print("      权重路径       = {}".format(path))
    if path is None:
        print("      !! 未找到权重，跳过")
        return

    if args.skip_model:
        print("      (--skip-model，不建模型)")
        return

    from detectron2.modeling import build_model
    from diffusiondet.weights import load_transfer_weights

    model = build_model(cfg)
    n_total = sum(p.numel() for p in model.parameters())
    info = load_transfer_weights(model, path=path, logger=None)
    if info is None:
        print("      !! 加载失败")
        return
    n_loaded = sum(p.numel() for k, p in model.named_parameters()
                   if k not in {s[0] for s in info["skipped_shape"]})
    print("      载入/总参数    = {:,} / {:,}".format(n_loaded, n_total))
    print("      形状不匹配被跳过 = {} 个 tensor".format(len(info["skipped_shape"])))
    for k, s_ckpt, s_model in info["skipped_shape"][:5]:
        print("        {:<64} {} -> {}".format(k, s_ckpt, s_model))
    print()
    print("SMOKE OK")


if __name__ == "__main__":
    main()
