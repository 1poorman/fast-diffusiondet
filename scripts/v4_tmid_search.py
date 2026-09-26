# V4: DPP@2 的 t_mid 网格搜索（docs/v2/BLUEPRINT.md，training-free）
# ========================================
# 背景：V0 证明轨迹噪端弯曲显著（43%），中间时间点的选择不再是无关变量；
# DPP@2 是唯一多步甜点（70.36）。本脚本固定 k=2 的 DPM-Solver++（degenerated），
# 把中间时间点从默认 499（均匀网格）扫到 {300..900}，找 AP 最优 t_mid。
#
# 运行（每卡一半网格）：
#   FASTDD_DATASET=$PWD/datasets CUDA_VISIBLE_DEVICES=0 \
#   python scripts/v4_tmid_search.py --ckpt output/lab/sdd.res50.bs8.long/model_final.pth \
#       --tmids 300,400,500,600 --out results/raw/v4_tmid_a.csv
import argparse
import csv
import contextlib
import io
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_net import Trainer, setup  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config-file", default="configs/lab/sdd.res50.bs8.long.yaml")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", default="results/raw/v4_tmid.csv")
    p.add_argument("--tmids", default="300,400,500,600,700,800,900")
    p.add_argument("--order", type=int, default=3)
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--eval-batch", type=int, default=32)
    p.add_argument("--tag", default="v4_tmid")
    return p.parse_args()


@torch.no_grad()
def dpp2_sample(detector, denoise_fn, batch, shape, batched_inputs, images, t_mid):
    """k=2 的 DPM-Solver++（degenerated）：与 run_sampler 内 dpm_v3 degenerated
    路径同构，仅时间网格换成 [(999, t_mid), (t_mid, -1)]。

    说明：直接调用 solvers.dpm_solver_v3 的 sample 接口需要重建 solver 与网格；
    k=2 的 DPM-Solver++(2 步、x_start 模型、logSNR skip) 数学上等价于
    「在 (t0→t_mid) 与 (t_mid→0) 两段各做一次 DDIM(eta=0) 确定性更新」，
    末步直达 x0。这里用后者实现以便精确控制 t_mid，且与 M2 的 ddim(eta=0)
    Euler 路径同源（M1 已验证两者一致）。
    """
    T = detector.num_timesteps
    time_pairs = [(T - 1, t_mid), (t_mid, -1)]
    img = torch.randn(shape, device=detector.device)

    for time, time_next in time_pairs:
        time_cond = torch.full((batch,), time, device=detector.device, dtype=torch.long)
        preds, outputs_class, outputs_coord = denoise_fn.call_model_predictions(img, time_cond)
        pred_noise, x_start = preds.pred_noise, preds.pred_x_start

        if time_next < 0:
            img = x_start
            continue
        alpha = detector.alphas_cumprod[time]
        alpha_next = detector.alphas_cumprod[time_next]
        # eta=0 确定性更新
        c = (1 - alpha_next).sqrt()
        img = x_start * alpha_next.sqrt() + c * pred_noise

    from diffusiondet.solvers.base import collect_results
    return collect_results(detector, batched_inputs, images, [], [], [],
                           denoise_fn.last_outputs_class, denoise_fn.last_outputs_coord)


def main():
    args = parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    tmids = [int(t) for t in args.tmids.split(",")]

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
    evaluator = Trainer.build_evaluator(cfg, cfg.DATASETS.TEST[0])

    from detectron2.evaluation import inference_on_dataset
    rows = []
    for t_mid in tmids:
        for seed in seeds:
            orig_forward = model.forward

            def custom_forward(inputs, do_postprocess=True, _tm=t_mid):
                feats = model.backbone(model.preprocess_image(inputs)[0].tensor)
                features = [feats[f] for f in model.in_features]
                images, images_whwh = model.preprocess_image(inputs)
                fn = model.make_denoise_fn(features, images_whwh)
                results = dpp2_sample(model, fn, images_whwh.shape[0],
                                      (images_whwh.shape[0], model.num_proposals, 4),
                                      inputs, images, _tm)
                return results

            torch.manual_seed(seed)
            with contextlib.redirect_stdout(io.StringIO()):
                results = inference_on_dataset(
                    custom_forward, loader, evaluator)
            model.forward = orig_forward

            ap = results["bbox"]["AP"]
            row = dict(tag=args.tag, t_mid=t_mid, seed=seed, AP=ap,
                       AP50=results["bbox"]["AP50"], AP75=results["bbox"]["AP75"])
            rows.append(row)
            print(f"[v4] {row}", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    # 汇总
    import statistics
    print("\n[v4] t_mid 汇总（AP mean±std over seeds）")
    for t_mid in tmids:
        aps = [r["AP"] for r in rows if r["t_mid"] == t_mid]
        print(f"  t_mid={t_mid}: {statistics.mean(aps)*100:.2f}±{statistics.stdev(aps)*100:.2f}"
              if len(aps) > 1 else f"  t_mid={t_mid}: {aps[0]*100:.2f}")
    print(f"[v4] saved -> {args.out}")


if __name__ == "__main__":
    main()
