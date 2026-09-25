# M4: 标定 EDM 的 sigma_data（D(x;s) 预条件需要）
# ========================================
# 蓝图 §3.3(A-2)：sigma_data = 训练集 prepare_diffusion_concat 输出 x_start 的
# 逐坐标标准差。x_start = GT 框（归一化 [-scale,scale] 的 cxcywh）+ 占位补齐
# （randn/6+0.5，detector.py:370-405 的噪声占位也是训练分布的一部分）。
# 产出：sigma_data.json（per-coord [cx,cy,w,h] 与 global 标量）。
#
# 运行：FASTDD_DATASET=$PWD/datasets python scripts/calib_sigma_data.py \
#   --config configs/lab/sdd.res50.yaml --out diffusiondet/sigma_data.json
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_net import Trainer, setup  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config-file", default="configs/lab/sdd.res50.yaml")
    p.add_argument("--out", default="diffusiondet/sigma_data.json")
    p.add_argument("--max-batches", type=int, default=200, help="训练 loader 批数")
    return p.parse_args()


def main():
    args = parse_args()

    class _Args:
        config_file = args.config_file
        eval_only = False
        num_gpus = 1
        num_machines = 1
        machine_rank = 0
        dist_url = "tcp://127.0.0.1:0"
        opts = []

    cfg = setup(_Args())
    cfg.defrost()
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.freeze()
    from diffusiondet.data_register import register_all
    register_all(cfg=cfg)

    model = Trainer.build_model(cfg)  # 只为拿 scale / num_proposals，不加载权重
    scale = model.scale
    P = model.num_proposals

    loader = Trainer.build_train_loader(cfg)
    gt_all = []
    for i, batch in enumerate(loader):
        for b in batch:
            inst = b["instances"]
            if len(inst) == 0:
                continue
            wh = torch.tensor([b["image"].shape[2], b["image"].shape[1]] * 2, dtype=torch.float32)
            bx = inst.gt_boxes.tensor  # xyxy abs
            cxcywh = torch.stack([(bx[:, 0] + bx[:, 2]) / 2, (bx[:, 1] + bx[:, 3]) / 2,
                                  bx[:, 2] - bx[:, 0], bx[:, 3] - bx[:, 1]], dim=1)
            cxcywh = (cxcywh / wh * 2 - 1.0) * scale
            gt_all.append(cxcywh)
        if i + 1 >= args.max_batches:
            break

    gt = torch.cat(gt_all)  # (M, 4)
    # 占位补齐框：训练时 x_start 里 (num_gt..P) 是 randn/6+0.5（detector.py:397），
    # 标准差 = 1/6，均值 0.5——与 GT 混合后整体 std 按训练时实际组成加权。
    # 训练时每图 num_gt 位置数随图变化；用数据集平均 GT 数近似。
    mean_gt_per_img = np.mean([len(g) for g in gt_all])
    frac_gt = min(mean_gt_per_img / P, 1.0)
    std_placeholder = 1.0 / 6.0  # randn/6+0.5 的 std（各坐标同）
    per_coord_std = gt.std(dim=0).numpy()  # GT 部分 per-coord
    # 混合分布 std：E[var] + var(E) 分解；占位与 GT 均值不同，需算总方差
    mu_gt = gt.mean(dim=0).numpy()
    var_total = frac_gt * (gt.var(dim=0).numpy() + 0.0) + (1 - frac_gt) * (std_placeholder ** 2) \
        + frac_gt * (1 - frac_gt) * (mu_gt - 0.5) ** 2  # 组间均值差贡献
    sigma_per_coord = np.sqrt(var_total)
    # global：所有坐标拼在一起的 RMS
    sigma_global = float(np.sqrt(np.mean(var_total)))

    out = {
        "scale": scale,
        "num_proposals": P,
        "frac_gt": float(frac_gt),
        "sigma_data_per_coord": sigma_per_coord.tolist(),  # [cx, cy, w, h]
        "sigma_data_global": sigma_global,
        "sigma_data_gt_only_per_coord": per_coord_std.tolist(),
        "sigma_data_gt_only_global": float(np.sqrt(np.mean(per_coord_std ** 2))),
        "note": ("prepare_diffusion_concat（训练用）含 randn/6+0.5 占位补齐，x_start 是"
                 "GT+占位混合分布 -> EDM sigma_data 取 sigma_data_global（混合版）。"
                 "gt_only 版仅作参考。per-coord order: cx, cy, w, h"),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
