# V0: box 域 ODE 轨迹直线性诊断（docs/v2/BLUEPRINT.md H5）
# ========================================
# 方法：固定 seed 的初始噪声出发，手动执行 DDIM(eta=0) k 步积分（eta=0 即 Euler
# 于 PF-ODE，与各 ODE solver 访问同一批时间点）。记录轨迹上每步的 x0 预测，
# 计算其相对终点的偏差：
#     rel_dev(t_i) = || x0_hat(t_i) - x0_final ||_2 / || x0_final ||_2
# 若全程（尤其噪端 t>=700）偏差很小 → 轨迹近似直线 → 求解器离散误差非主导，
# H5 支持，"换求解器"路线永久关闭。
#
# 运行：
#   FASTDD_DATASET=$PWD/datasets CUDA_VISIBLE_DEVICES=1 \
#   python scripts/diag_trajectory.py \
#       --config-file configs/lab/sdd.res50.bs8.long.yaml \
#       --ckpt output/lab/sdd.res50.bs8.long/model_final.pth \
#       --out results/raw/v2_traj_diag.csv
import argparse
import csv
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_net import Trainer, setup  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config-file", default="configs/lab/sdd.res50.bs8.long.yaml")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", default="results/raw/v2_traj_diag.csv")
    p.add_argument("--steps", type=int, default=10, help="ODE 离散步数 k")
    p.add_argument("--eval-batch", type=int, default=32)
    p.add_argument("--num-batches", type=int, default=0, help="0 = 全 val 集")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


@torch.no_grad()
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
    cfg.MODEL.WEIGHTS = args.ckpt
    cfg.MODEL.DiffusionDet.BOX_RENEWAL = False
    cfg.MODEL.DiffusionDet.USE_ENSEMBLE = False
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.freeze()
    from diffusiondet.data_register import register_all
    register_all(cfg=cfg)

    model = Trainer.build_model(cfg)
    from detectron2.checkpoint import DetectionCheckpointer
    DetectionCheckpointer(model, save_dir="").resume_or_load(cfg.MODEL.WEIGHTS, resume=False)
    model.eval()

    loader = Trainer.build_test_loader(cfg, cfg.DATASETS.TEST[0])

    k = args.steps
    T = model.num_timesteps
    # 与 ddim_sample 相同的网格：均匀 linspace 的倒序 time_pairs
    times = torch.linspace(-1, T - 1, steps=k + 1)
    times = list(reversed(times.int().tolist()))
    time_pairs = list(zip(times[:-1], times[1:]))
    assert time_pairs[-1][1] < 0, "末步必须直达 x0（time_next=-1）"
    n_t = len(time_pairs)

    dev_sum = [0.0] * n_t
    dev_sq_sum = [0.0] * n_t
    dev_max = [0.0] * n_t
    n_img = 0
    done_batches = 0

    for batched_inputs in loader:
        if args.num_batches and done_batches >= args.num_batches:
            break
        done_batches += 1
        torch.manual_seed(args.seed)  # 每批同 seed 起点，诊断可复现

        # 与 detector.forward 推理分支完全一致的特征提取
        images, images_whwh = model.preprocess_image(batched_inputs)
        src = model.backbone(images.tensor)
        features = [src[f] for f in model.in_features]

        batch = images_whwh.shape[0]
        img = torch.randn((batch, model.num_proposals, 4), device=model.device)
        x0_hat_seq = []
        x0_final = None
        for time, time_next in time_pairs:
            time_cond = torch.full((batch,), time, device=model.device, dtype=torch.long)
            preds, _, _ = model.model_predictions(features, images_whwh, img, time_cond, None, True)
            pred_noise, x_start = preds.pred_noise, preds.pred_x_start
            x0_hat_seq.append(x_start)
            if time_next < 0:
                x0_final = x_start
                continue
            alpha = model.alphas_cumprod[time]
            alpha_next = model.alphas_cumprod[time_next]
            # eta=0：DDIM 确定性更新 = w=σ/α 参数化下的 Euler
            c = (1 - alpha_next).sqrt()
            img = x_start * alpha_next.sqrt() + c * pred_noise

        # per-image 相对偏差（proposal+coord 联合 L2）
        denom = x0_final.flatten(1).norm(dim=1)  # (B,)
        for i, x0_hat in enumerate(x0_hat_seq):
            dev = (x0_hat - x0_final).flatten(1).norm(dim=1) / denom.clamp(min=1e-8)  # (B,)
            dev_sum[i] += dev.sum().item()
            dev_sq_sum[i] += (dev ** 2).sum().item()
            dev_max[i] = max(dev_max[i], dev.max().item())
        n_img += batch

    rows = []
    print(f"\n[V0] 轨迹直线性（k={k}, images={n_img}, eta=0, seed={args.seed}）")
    print(f"{'t':>6} {'rel_dev_mean%':>14} {'rel_dev_std%':>13} {'rel_dev_max%':>13}")
    for i, (time, _) in enumerate(time_pairs):
        mean = 100.0 * dev_sum[i] / n_img
        var = max(dev_sq_sum[i] / n_img - (dev_sum[i] / n_img) ** 2, 0.0)
        std = 100.0 * var ** 0.5
        mx = 100.0 * dev_max[i]
        print(f"{time:>6} {mean:>14.3f} {std:>13.3f} {mx:>13.3f}")
        rows.append(dict(tag="v0_traj", k=k, seed=args.seed, t=time,
                         rel_dev_mean_pct=round(mean, 4),
                         rel_dev_std_pct=round(std, 4),
                         rel_dev_max_pct=round(mx, 4), images=n_img))

    # 汇总判据：噪端（前 1/3 步，t 最大段）平均偏差
    n_noise = max(1, n_t // 3)
    noise_end = sum(r["rel_dev_mean_pct"] for r in rows[:n_noise]) / n_noise
    print(f"\n[V0] 噪端平均相对偏差 = {noise_end:.3f}%")
    if noise_end < 10.0:
        print("[V0] H5 判定倾向：支持（轨迹近似直线，求解器离散误差非主导）")
    elif noise_end > 20.0:
        print("[V0] H5 判定倾向：存疑（噪端轨迹弯曲显著，V4 时间网格值得做）")
    else:
        print("[V0] H5 判定倾向：边界，需结合 V1/V4 结果综合判断")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[V0] saved -> {args.out}")


if __name__ == "__main__":
    main()
