# M2: Training-free 对比矩阵扫描（E0-{DDIM,EULER,HEUN,DPP} × NFE × seed）
# ========================================
# 蓝图 §M2：
#   * 所有 arm 共用同一个 baseline checkpoint（权重冻结）；
#   * 主矩阵在 BOX_RENEWAL=False + USE_ENSEMBLE=False 下跑（纯采样器曲线）；
#   * NFE 预算网格 {1,2,3,4,6,8,10}；Heun 的 step 数 k=(B+1)//2（实际 NFE=2k-1，如实记录）；
#   * 每个配置 3 seed，AP 取均值±std；延迟为 eval 期间逐图 forward 计时（中位数）。
#
# 运行示例：
#   FASTDD_DATASET=$PWD/datasets CUDA_VISIBLE_DEVICES=1 \
#   python scripts/run_m2_matrix.py --ckpt output/lab/sdd.res50/model_final.pth \
#       --out results/raw/m2_matrix.csv
import argparse
import contextlib
import csv
import io
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_net import Trainer, setup  # noqa: E402

SOLVERS = ["ddim", "euler", "heun", "dpm_v3"]
NFE_BUDGETS = [1, 2, 3, 4, 6, 8, 10]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config-file", default="configs/lab/sdd.res50.yaml")
    p.add_argument("--ckpt", required=True, help="baseline checkpoint (所有 arm 共用)")
    p.add_argument("--out", default="results/raw/m2_matrix.csv")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--solvers", default=",".join(SOLVERS))
    p.add_argument("--nfes", default=",".join(map(str, NFE_BUDGETS)))
    p.add_argument("--eval-batch", type=int, default=8, help="评测 batch（不影响 AP，只影响速度）")
    p.add_argument("--renewal", type=int, default=0, choices=[0, 1],
                   help="A1 消融：1 开启 box renewal（仅 DDIM 支持）")
    p.add_argument("--ensemble", type=int, default=0, choices=[0, 1],
                   help="A2 消融：1 开启多步 ensemble + NMS（仅 DDIM 支持）")
    p.add_argument("--tag", default="m2", help="输出行标签，区分主矩阵/消融")
    p.add_argument("--dataset", default="sdd_val", help="评测数据集（pls_val 用于 M6 确认）")
    p.add_argument("--stats-dir", default="", help="DPMv3 EMS 统计量目录（M3：非 degenerated）")
    p.add_argument("--per-coord", action="store_true", help="使用 per_coord 统计量")
    return p.parse_args()


def steps_for_budget(solver, budget):
    """NFE 预算 -> (sampling_timesteps, 实际 NFE)。"""
    if solver in ("ddim", "euler", "dpm_v3"):
        return budget, budget
    if solver == "heun":
        k = (budget + 1) // 2  # NFE = 2k-1
        return k, 2 * k - 1
    raise ValueError(solver)


@contextlib.contextmanager
def patch_model(model, solver, k):
    """临时改写 forward 期读取的采样器属性（均在 __init__ 赋值、forward 读取）。

    renewal / ensemble / degenerated 等由 cfg（main 里）控制，此处只切
    solver 与步数，保持消融语义清晰。
    """
    saved = (model.solver_name, model.sampling_timesteps)
    model.solver_name = solver
    model.sampling_timesteps = k
    yield
    (model.solver_name, model.sampling_timesteps) = saved


def main():
    args = parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    solvers = args.solvers.split(",")
    budgets = [int(b) for b in args.nfes.split(",")]

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
    cfg.MODEL.DiffusionDet.BOX_RENEWAL = bool(args.renewal)
    cfg.MODEL.DiffusionDet.USE_ENSEMBLE = bool(args.ensemble)
    if args.stats_dir:
        cfg.MODEL.DiffusionDet.DEGENERATED = False
        cfg.MODEL.DiffusionDet.STATS_DIR = args.stats_dir
    if args.renewal or args.ensemble:
        # A1/A2 消融是 DDIM 专属机制；ODE solver 不支持，强制只用 ddim
        assert args.solvers == "ddim", "renewal/ensemble ablation only supports --solvers ddim"
    cfg.DATASETS.TEST = (args.dataset,)
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.TEST.EVAL_PERIOD = 0
    cfg.freeze()
    from diffusiondet.data_register import register_all
    register_all(cfg=cfg)

    model = Trainer.build_model(cfg)
    from detectron2.checkpoint import DetectionCheckpointer
    DetectionCheckpointer(model, save_dir="").resume_or_load(cfg.MODEL.WEIGHTS, resume=False)
    model.eval()

    loader = Trainer.build_test_loader(cfg, cfg.DATASETS.TEST[0])
    evaluator = Trainer.build_evaluator(cfg, cfg.DATASETS.TEST[0])

    rows = []
    seen = set()  # 同 (solver, k, seed) 只跑一次（heun 预算 1/2 都映射 k=1 等）
    for solver in solvers:
        for budget in budgets:
            k, actual_nfe = steps_for_budget(solver, budget)
            if (solver, k) in seen and len(seeds) > 0:
                continue
            seen.add((solver, k))
            for seed in seeds:
                t0 = time.perf_counter()
                # 标准评测流程：inference_on_dataset 负责喂预测给 evaluator
                from detectron2.evaluation import inference_on_dataset

                with patch_model(model, solver, k):
                    torch.manual_seed(seed)
                    lat = []
                    orig_forward = model.forward

                    def timed_forward(*a, _orig=orig_forward, **kw):
                        t1 = time.perf_counter()
                        r = _orig(*a, **kw)
                        torch.cuda.synchronize()
                        lat.append(time.perf_counter() - t1)
                        return r

                    model.forward = timed_forward
                    with contextlib.redirect_stdout(io.StringIO()):
                        results = inference_on_dataset(model, loader, evaluator)
                    model.forward = orig_forward

                lat_ms = sorted(lat)
                median_ms = 1000 * lat_ms[len(lat_ms) // 2]
                mean_ms = 1000 * sum(lat) / len(lat)
                ap = results["bbox"]["AP"]
                ap50 = results["bbox"]["AP50"]
                ap75 = results["bbox"]["AP75"]
                row = dict(tag=args.tag, solver=solver, nfe_budget=budget, sampling_steps=k,
                           actual_nfe=actual_nfe, seed=seed, AP=ap, AP50=ap50, AP75=ap75,
                           latency_ms_median=round(median_ms, 2),
                           latency_ms_mean=round(mean_ms, 2),
                           wall_s=round(time.perf_counter() - t0, 1))
                rows.append(row)
                print(f"[m2] {row}", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[m2] saved -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
