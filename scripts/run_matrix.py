# -*- coding: utf-8 -*-
"""
M2 training-free 对比矩阵。

同一个 baseline checkpoint，权重完全冻结，只换推理侧的采样器，
扫 AP - NFE 与 latency - NFE 曲线。

用法（按 solver 分块跑，避免单次运行过久）：
    python scripts/run_matrix.py --solvers ddim   --seeds 0 1 2
    python scripts/run_matrix.py --solvers euler  --seeds 0 1 2
    python scripts/run_matrix.py --solvers heun   --seeds 0 1 2
    python scripts/run_matrix.py --solvers dpm_solver_v3 --seeds 0 1 2 [--dpm-order 2]

结果追加写入 results/raw/matrix.csv。
"""

import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.modeling import build_model
from detectron2.utils.logger import setup_logger

from diffusiondet import add_diffusiondet_config, DiffusionDetDatasetMapper
from diffusiondet.data_register import register_all
from diffusiondet.util.model_ema import add_model_ema_configs

# solver -> [(param, expected_nfe), ...]
# NOTE: 必须用全量 split 评测（见 make_subset 的注释），因此网格刻意保持在
# 低 NFE 区域 —— 那里才是本项目立论所在（NFE<=4）。
NFE_GRID = {
    "ddim": [(1, 1), (2, 2), (4, 4)],
    "euler": [(1, 1), (2, 2), (4, 4)],
    # heun: NFE = 2*n - 1
    "heun": [(1, 1), (2, 3), (3, 5)],
    "dpm_solver_v3": [(1, 1), (2, 2), (4, 4)],
}


def make_subset(name, src, n):
    """
    ⚠ **不要用**：注册只含前 n 张图的评测子集来"加速扫描"是**无效**的。

    实测（SDD val，DDIM NFE=1）：
        n=20  -> AP  2.02
        n=250 -> AP 21.92
        n=750 -> AP 60.53
    AP 与子集大小近似成正比 —— 说明 pycocotools 的召回分母仍按**全量 GT** 计算，
    只在前 n 张图上推理会系统性低估 AP。
    保留此函数仅供复现该结论，**跑矩阵时必须用全量 split（--num-images 0）**。
    """
    dicts = DatasetCatalog.get(src)[:n]
    DatasetCatalog.register(name, lambda d=dicts: d)
    src_meta = MetadataCatalog.get(src)
    MetadataCatalog.get(name).set(
        thing_classes=list(src_meta.thing_classes),
        thing_dataset_id_to_contiguous_id=dict(src_meta.thing_dataset_id_to_contiguous_id),
        json_file=src_meta.json_file,
        image_root=src_meta.image_root,
        evaluator_type="coco",
    )
    return name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-file", default="configs/lab/sdd.res50.yaml")
    ap.add_argument("--weights", default="output/lab/sdd.res50/model_final.pth")
    ap.add_argument("--split", default="sdd_val")
    ap.add_argument("--num-images", type=int, default=250, help="0 = 全量")
    ap.add_argument("--solvers", nargs="+", default=["ddim"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--box-renewal", type=int, default=0)
    ap.add_argument("--use-ensemble", type=int, default=0)
    ap.add_argument("--dpm-order", type=int, default=2)
    ap.add_argument("--out-csv", default="results/raw/matrix.csv")
    args = ap.parse_args()
    setup_logger()

    base_cfg = get_cfg()
    add_diffusiondet_config(base_cfg)
    add_model_ema_configs(base_cfg)
    base_cfg.merge_from_file(args.config_file)
    register_all(cfg=base_cfg)

    eval_split = args.split
    if args.num_images > 0:
        eval_split = make_subset("{}_sub{}".format(args.split, args.num_images),
                                 args.split, args.num_images)

    cfg = base_cfg.clone()
    cfg.defrost()
    cfg.MODEL.DEVICE = "cuda"
    cfg.MODEL.DiffusionDet.DPM_ORDER = args.dpm_order
    cfg.freeze()

    model = build_model(cfg)
    DetectionCheckpointer(model).load(args.weights)
    model.eval().to("cuda")
    model.box_renewal = bool(args.box_renewal)
    model.use_ensemble = bool(args.use_ensemble)

    loader = build_detection_test_loader(
        cfg, eval_split, mapper=DiffusionDetDatasetMapper(cfg, is_train=False))

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    new = not os.path.isfile(args.out_csv)
    fout = open(args.out_csv, "a", newline="", encoding="utf-8")
    w = csv.writer(fout)
    if new:
        w.writerow(["tag", "dataset", "split", "n_images", "solver", "param", "nfe", "seed",
                    "AP", "AP50", "AP75", "box_renewal", "use_ensemble", "eval_sec"])

    for solver in args.solvers:
        for param, nfe in NFE_GRID[solver]:
            for seed in args.seeds:
                model.solver_name = solver
                model.sampling_timesteps = param
                torch.manual_seed(seed)
                torch.cuda.empty_cache()

                evaluator = COCOEvaluator(eval_split, cfg, False,
                                          output_dir=os.path.join("output", "matrix_tmp"))
                t0 = time.time()
                res = inference_on_dataset(model, loader, evaluator)
                dt = time.time() - t0
                bbox = res.get("bbox", {})
                ap_v = bbox.get("AP", float("nan"))
                w.writerow(["sdd.f0.{}".format(solver), "sdd", eval_split,
                            len(DatasetCatalog.get(eval_split)), solver, param, nfe, seed,
                            round(ap_v, 4), round(bbox.get("AP50", float("nan")), 4),
                            round(bbox.get("AP75", float("nan")), 4),
                            int(args.box_renewal), int(args.use_ensemble), round(dt, 1)])
                fout.flush()
                print("[matrix] {:<14} param={:<3} nfe={:<3} seed={}  AP={:.4f}  AP50={:.4f}  ({:.0f}s)".format(
                    solver, param, nfe, seed, ap_v, bbox.get("AP50", float("nan")), dt))

    fout.close()
    print("[run_matrix] -> {}".format(args.out_csv))


if __name__ == "__main__":
    main()
