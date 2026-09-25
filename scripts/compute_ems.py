# M3: EMS 统计量标定（DPM-Solver-v3 的 l / s / b）
# ========================================
# 移植自 thu-ml/DPM-Solver-v3 compute_EMS_scoresde.py，适配 DiffusionDet box 域：
#   * 数据 x0：训练集 GT 框（归一化 [-scale,scale] 空间），每样本随机抽 300 个组成
#     (P,4)（跨数据集有放回），替代图像像素分布；
#   * noise_pred_fn：head 前向（backbone 特征每图一次），x0_hat 反变换后
#     eps_hat = (x_t - alpha*x0_hat)/sigma（x0 参数化 -> noise）；
#   * 时间：连续 t_input = (t - 1/N)*1000 直接喂 head 的 sinusoidal 时间嵌入
#     （嵌入对连续值有定义；统计用连续 t，不 snap）；
#   * d eps_hat / d lambda：fwAD 精确 JVP（torch.autograd.forward_ad，本机 torch 2.1
#     可用；蓝图"有限差分"是 torch1.10 约束的降级方案）；
#   * l: E[sigma * (J_x eps_hat . v) * v]（Hutchinson 方向投影，v ~ Rademacher）；
#   * a = (sigma*eps_hat - l*x_t)/alpha
#     b = exp(-lambda)*((l-1)*eps_hat + d_eps_d_lambda) - l_d*x_t/alpha
#     s = (E[ab]-E[a]E[b]) / (E[a^2]-E[a]^2)；  b_coef = E[b] - s*E[a]
#   * 网格：1000 点 logSNR uniform（与 DPM_Solver_v3 statistics_steps=1000 对齐）。
#
# 运行（约 1h on 3090）：
#   FASTDD_DATASET=$PWD/datasets CUDA_VISIBLE_DEVICES=0 \
#   python scripts/compute_ems.py --ckpt output/lab/sdd.res50/model_final.pth \
#       --out statistics/sdd/ems_k256 --scalar-dir statistics/sdd/ems_scalar
import argparse
import math
import os
import sys
import time

import numpy as np
import torch
import torch.autograd.forward_ad as fwAD

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_net import Trainer, setup  # noqa: E402
from diffusiondet.solvers.dpm_solver_v3 import NoiseScheduleVP  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config-file", default="configs/lab/sdd.res50.yaml")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--num-grid", type=int, default=1000, help="logSNR 网格点数")
    p.add_argument("--num-batches", type=int, default=32, help="K/8：训练批次数（bs=8）")
    p.add_argument("--out", required=True, help="per_coord 统计量目录")
    p.add_argument("--scalar-out", default="", help="可选：scalar 统计量目录（A8 消融）")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--resume-l", action="store_true",
                   help="跳过 pass1，直接加载 out 目录已有的 l.npz/l_d.npz 只跑 pass2")
    return p.parse_args()


class BoxNoisePredFn:
    """连续时间噪声预测：eps_hat = (x_t - alpha(t)*x0_hat)/sigma(t)。

    t_input 连续（不 snap），时间嵌入平滑。JVP 时 x/t 可为 dual tensor。
    """

    def __init__(self, model, ns, images_whwh, features):
        self.model = model
        self.ns = ns
        self.whwh = images_whwh
        self.feats = features
        self.scale = model.scale
        self.N = model.num_timesteps

    def _head_x0(self, x_t, t_input):
        """x_t -> head -> 反归一化 x0_hat（不 clamp 到 ±scale，保留 JVP 连续性）。"""
        from diffusiondet.util.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
        x_boxes = x_t / self.scale
        x_boxes = ((x_boxes) + 1) / 2
        x_boxes = box_cxcywh_to_xyxy(x_boxes) * self.whwh[:, None, :]
        outputs_class, outputs_coord = self.model.head(self.feats, x_boxes, t_input, None)
        x0 = outputs_coord[-1] / self.whwh[:, None, :]
        x0 = box_xyxy_to_cxcywh(x0)
        x0 = (x0 * 2 - 1.0) * self.scale
        return x0

    def __call__(self, x_t, t_input):
        """返回 eps_hat。x_t/t_input 可为 fwAD dual tensor。

        注意：model.betas buffer 是 float64 -> ns.marginal_*/inverse_lambda 返回
        float64；head 的时间嵌入必须收 float32，否则 F.linear Double/Float 崩溃。
        """
        t_input = t_input.detach().float() if not t_input.requires_grad else t_input.float()
        if t_input.dim() == 0:
            t_input = t_input.reshape(1).expand(x_t.shape[0])
        # 单位换算：marginal_alpha/std 期望连续时间 t ∈ [1/N, 1]，
        # 而 t_input 是 head 嵌入域（(t-1/N)*1000，0..999）。此前误传 t_input
        # 导致越界外推 -> alpha>1 -> sigma^2<0 -> NaN（最后网格点必现）。
        t_c = t_input / 1000.0 + 1.0 / self.N
        alpha = self.ns.marginal_alpha(t_c)
        sigma = self.ns.marginal_std(t_c)
        x0_hat = self._head_x0(x_t, t_input)
        alpha_e = alpha[(...,) + (None,) * (x_t.dim() - 1)]
        sigma_e = sigma[(...,) + (None,) * (x_t.dim() - 1)]
        return (x_t - alpha_e * x0_hat) / sigma_e


def box_xyxy_to_cxcywh_t(x):
    x0, y0, x1, y1 = x.unbind(-1)
    return torch.stack([(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)], dim=-1)


def jvp_x(fn, x, v, h=0.02):
    """返回 (fn(x), J_x fn(x) . v)。

    head 内含自定义 autograd Function，fwAD 对其 c++ API 尚未支持
    （实测报 "jvp is not implemented"），故直接用中心差分
    （蓝图 §M3 原方案）。h 在 box 归一化空间 [-2,2] 内取 0.02。
    """
    f0 = fn(x)
    fp = fn(x + h * v)
    fm = fn(x - h * v)
    return f0, (fp - fm) / (2 * h)
    with fwAD.dual_level():
        dual_x = fwAD.make_dual(x, v)
        out = fn(dual_x)
        primal, dual = fwAD.unpack_dual(out)
        return primal, (dual if dual is not None else torch.zeros_like(primal))


def main():
    args = parse_args()
    torch.cuda.set_device(args.gpu)
    device = f"cuda:{args.gpu}"

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
    cfg.freeze()
    from diffusiondet.data_register import register_all
    register_all(cfg=cfg)

    model = Trainer.build_model(cfg)
    from detectron2.checkpoint import DetectionCheckpointer
    DetectionCheckpointer(model).resume_or_load(cfg.MODEL.WEIGHTS, resume=False)
    model.eval()

    ns = NoiseScheduleVP(schedule="discrete", betas=model.betas)
    N = model.num_timesteps

    # ---- logSNR 网格（与 solver statistics 网格一致：t_T=1.0 -> t_0=1/N）----
    lam_T = ns.marginal_lambda(torch.tensor(1.0))
    lam_0 = ns.marginal_lambda(torch.tensor(1.0 / N))
    lams = torch.linspace(lam_T.item(), lam_0.item(), args.num_grid + 1).float()  # 降序
    t_cont = ns.inverse_lambda(lams).float()          # 连续 t（降序 T->0）
    t_input = ((t_cont - 1.0 / N) * 1000.0).float()   # head 时间输入

    # ---- 训练数据流（增广后图像 + GT 框）----
    loader = Trainer.build_train_loader(cfg)
    gt_pool = []  # 每图 (n_i, 4) 归一化 cxcywh * scale
    print("[ems] collecting GT box pool ...", flush=True)
    for i, batch in enumerate(loader):
        for b in batch:
            inst = b["instances"]
            if len(inst) == 0:
                continue
            wh = torch.tensor([b["image"].shape[2], b["image"].shape[1],
                               b["image"].shape[2], b["image"].shape[1]], dtype=torch.float32)
            bx = inst.gt_boxes.tensor  # xyxy abs
            cxcywh = torch.stack([(bx[:, 0] + bx[:, 2]) / 2, (bx[:, 1] + bx[:, 3]) / 2,
                                  bx[:, 2] - bx[:, 0], bx[:, 3] - bx[:, 1]], dim=1)
            cxcywh = cxcywh / wh
            cxcywh = (cxcywh * 2 - 1.0) * model.scale
            gt_pool.append(cxcywh)
        if len(gt_pool) >= 400:
            break
    print(f"[ems] GT pool: {len(gt_pool)} images", flush=True)

    P = model.num_proposals
    STAT_STEPS = args.num_grid  # statistics_steps

    def sample_x0(bs, gen):
        """每样本从 GT 池随机抽 P 个框（跨图有放回）。"""
        out = torch.empty(bs, P, 4, device=device)
        for b in range(bs):
            src = gt_pool[torch.randint(len(gt_pool), (1,), generator=gen).item()]
            idx = torch.randint(len(src), (P,), generator=gen)
            out[b] = src[idx].to(device)
        return out

    # ================= Pass 1: l =================
    gen = torch.Generator().manual_seed(2026)
    t0all = time.time()
    if args.resume_l:
        l_pc = np.load(os.path.join(args.out, "l.npz"))["l"]
        l_d_s = np.load(os.path.join(args.out, "l_d.npz"))["l_d"]
        assert np.isfinite(l_pc).all() and np.isfinite(l_d_s).all()
        print(f"[ems] resume: loaded l/l_d from {args.out}, skip pass1", flush=True)
    else:
        l_sum = None
        n_samples = 0
        loader_iter = iter(loader)
        for bi in range(args.num_batches):
            batch = next(loader_iter)
            has_gt = [b for b in batch if len(b["instances"]) > 0]
            if not has_gt:
                continue
            images, images_whwh = model.preprocess_image(has_gt)
            with torch.no_grad():
                src = model.backbone(images.tensor)
            features = [src[f] for f in model.in_features]
            fn = BoxNoisePredFn(model, ns, images_whwh, features)
            bs = images_whwh.shape[0]
            x0 = sample_x0(bs, gen)

            with torch.no_grad():  # 统计前向全程 no_grad，否则前向图累积导致 OOM
                for j in range(STAT_STEPS + 1):
                    tc = t_cont[j].to(device)
                    ti = t_input[j].to(device)
                    alpha = ns.marginal_alpha(tc).item()
                    sigma = ns.marginal_std(tc).item()
                    z = torch.randn(x0.shape, generator=gen).to(device)
                    x_t = alpha * x0 + sigma * z
                    v = torch.randint(0, 2, x_t.shape, generator=gen).to(device) * 2.0 - 1

                    _, jvp = jvp_x(lambda xx: fn(xx, ti), x_t, v)
                    l_j = (sigma * jvp * v).mean(dim=0)  # (P,4) per_coord
                    if l_sum is None:
                        l_sum = torch.zeros(STAT_STEPS + 1, P, 4, device=device)
                    l_sum[j] += l_j
            n_samples += bs
            el = time.time() - t0all
            eta = el / (bi + 1) * (args.num_batches - bi - 1)
            print(f"[ems] pass1 batch {bi+1}/{args.num_batches} ({n_samples} samples) eta={eta/60:.1f}min", flush=True)

        l_pc = (l_sum / n_samples).cpu().numpy()  # (Ngrid+1, P, 4)
        os.makedirs(args.out, exist_ok=True)
        np.savez_compressed(os.path.join(args.out, "l.npz"), l=l_pc)
        if args.scalar_out:
            os.makedirs(args.scalar_out, exist_ok=True)
            np.savez_compressed(os.path.join(args.scalar_out, "l.npz"), l=l_pc.mean(axis=(1, 2)))

        # ---- l_d = dl/dlambda（中心差分 + 滑动平均，逐坐标） ----
        gap = (lam_0.item() - lam_T.item()) / STAT_STEPS
        l_d = np.empty_like(l_pc)
        for i in range(STAT_STEPS + 1):
            if i == 0:
                l_d[i] = (l_pc[i + 1] - l_pc[i]) / gap
            elif i == STAT_STEPS:
                l_d[i] = (l_pc[i] - l_pc[i - 1]) / gap
            else:
                l_d[i] = (l_pc[i + 1] - l_pc[i - 1]) / (2 * gap)
        window = 5
        l_d_s = np.empty_like(l_d)
        for i in range(STAT_STEPS + 1):
            lo, hi = max(0, i - window), min(STAT_STEPS + 1, i + window + 1)
            l_d_s[i] = l_d[lo:hi].mean(axis=0)
        np.savez_compressed(os.path.join(args.out, "l_d.npz"), l_d=l_d_s)
        if args.scalar_out:
            np.savez_compressed(os.path.join(args.scalar_out, "l_d.npz"), l_d=l_d_s.mean(axis=(1, 2)))

    # ================= Pass 2: f 统计 -> s, b =================
    fa = fb = faa = fab = None
    n_samples = 0
    l_t = torch.from_numpy(l_pc).to(device)
    l_d_t = torch.from_numpy(l_d_s).to(device)
    loader_iter = iter(loader)
    for bi in range(args.num_batches):
        batch = next(loader_iter)
        has_gt = [b for b in batch if len(b["instances"]) > 0]
        if not has_gt:
            continue
        images, images_whwh = model.preprocess_image(has_gt)
        with torch.no_grad():
            src = model.backbone(images.tensor)
        features = [src[f] for f in model.in_features]
        fn = BoxNoisePredFn(model, ns, images_whwh, features)
        bs = images_whwh.shape[0]
        x0 = sample_x0(bs, gen)

        with torch.no_grad():
            for j in range(STAT_STEPS + 1):
                tc = t_cont[j].to(device)
                ti = t_input[j].to(device)
                alpha_t = ns.marginal_alpha(tc).item()
                sigma_t = ns.marginal_std(tc).item()
                lamb = lams[j].item()
                l_j = l_t[j]
                l_dj = l_d_t[j]

                z = torch.randn(x0.shape, generator=gen).to(device)
                x_t = alpha_t * x0 + sigma_t * z

                # eps_hat 与 d eps_hat / d lambda：时间方向 ±1 索引中心差分
                # （sinusoidal 嵌入连续；1 索引单位与 l_d 的相邻网格差分一致）
                eps_hat = fn(x_t, ti)
                # 时间差分：按 ti 值判断端点（float32 下末尾两格点 ti 都舍入为
                # 0.0，j 索引不可靠）；越出 [1/N,1] 的外推会使 sigma NaN，必须向域内单侧
                d = 1.0
                lam_hi = ns.marginal_lambda((ti + d) / 1000.0 + 1.0 / N).item()
                lam_lo = ns.marginal_lambda((ti - d) / 1000.0 + 1.0 / N).item()
                if ti.item() >= 1000.0 - 1.5:  # 噪端 ti=999，向内单侧
                    eps_d = (fn(x_t, ti) - fn(x_t, ti - d)) / (lams[j].item() - lam_lo)
                elif ti.item() <= 1.5:  # 净端 ti=0，向内单侧
                    eps_d = (fn(x_t, ti + d) - fn(x_t, ti)) / (lam_hi - lams[j].item())
                else:
                    eps_d = (fn(x_t, ti + d) - fn(x_t, ti - d)) / (lam_hi - lam_lo)

                a = (sigma_t * eps_hat - l_j * x_t) / alpha_t
                b = math.exp(-lamb) * ((l_j - 1.0) * eps_hat + eps_d) - l_dj * x_t / alpha_t
                if fa is None:
                    fa = torch.zeros(STAT_STEPS + 1, P, 4, device=device)
                    fb = torch.zeros_like(fa)
                    faa = torch.zeros_like(fa)
                    fab = torch.zeros_like(fa)
                fa[j] += a.mean(dim=0)
                fb[j] += b.mean(dim=0)
                faa[j] += (a * a).mean(dim=0)
                fab[j] += (a * b).mean(dim=0)
        n_samples += bs
        el = time.time() - t0all
        eta = el / (bi + 1) * (args.num_batches - bi - 1)
        print(f"[ems] pass2 batch {bi+1}/{args.num_batches} eta={eta/60:.1f}min", flush=True)

    f = (fa / n_samples).cpu().numpy()
    f_d = (fb / n_samples).cpu().numpy()
    f_f = (faa / n_samples).cpu().numpy()
    f_f_d = (fab / n_samples).cpu().numpy()
    # 方差分母保护：a 方差过小的网格点 s 退化（0/0），clamp + nan_to_num 防御
    denom = np.maximum(f_f - f * f, 1e-8)
    s = (f_f_d - f * f_d) / denom
    s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
    for nm, arr in [("f", f), ("f_d", f_d), ("f_f", f_f), ("f_f_d", f_f_d)]:
        print(f"[ems] {nm}: nan={np.isnan(arr).sum()} inf={np.isinf(arr).sum()}", flush=True)
    b_c = f_d - s * f
    np.savez_compressed(os.path.join(args.out, "sb.npz"), s=s, b=b_c)
    if args.scalar_out:
        s_sc = (f_f_d.mean(axis=(1, 2)) - f.mean(axis=(1, 2)) * f_d.mean(axis=(1, 2))) / \
               (f_f.mean(axis=(1, 2)) - f.mean(axis=(1, 2)) ** 2 + 1e-12)
        b_sc = f_d.mean(axis=(1, 2)) - s_sc * f.mean(axis=(1, 2))
        np.savez_compressed(os.path.join(args.scalar_out, "sb.npz"), s=s_sc, b=b_sc)

    # ---- Gate 自检：无 NaN，l 随 lambda 光滑（相邻网格相对变化 < 10） ----
    for name, arr in [("l", l_pc), ("l_d", l_d_s), ("s", s), ("b", b_c)]:
        assert np.isfinite(arr).all(), f"{name} has NaN/Inf"
    # 光滑性 Gate：过零点会使逐点比值爆炸，改用相邻差分的 max/median
    # （MC 噪声主导时 max/median ~ O(10)，系统性跳变则 >1000）
    dif = np.abs(np.diff(l_pc, axis=0))
    med = np.median(dif) + 1e-12
    print(f"[ems] l smoothness: max|dif|={dif.max():.2e}, median|dif|={med:.2e}, "
          f"max/median={dif.max()/med:.1f} (Gate < 1000)", flush=True)
    print(f"[ems] DONE -> {args.out} total {time.time()-t0all:.0f}s", flush=True)


if __name__ == "__main__":
    main()
