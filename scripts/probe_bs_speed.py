# -*- coding: utf-8 -*-
"""bs 搜索：在 24GB 显存内找「吞吐最大」的训练 batch，并实测 s/iter 与峰值显存。

用法：
    FASTDD_DATASET=$PWD/datasets CUDA_VISIBLE_DEVICES=1 \
    ~/.conda/envs/fastdiff/bin/python scripts/probe_bs_speed.py \
        --bs 24 [--iters 30] [--opts KEY VAL ...]

说明：
  * 只改 SOLVER.IMS_PER_BATCH 与 LR（默认按 linear rule = 1e-4 * bs/8），
    不改模型结构，配置与主矩阵同源；
  * 显存随输入尺寸波动（MIN_SIZE_TRAIN 是 480~800 的随机采样），
    所以峰值要跑够迭代数才可信，留够安全余量再上生产训练。
"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config-file", default="configs/lab/sdd.res50.bs8.yaml")
    p.add_argument("--bs", type=int, required=True)
    p.add_argument("--iters", type=int, default=30, help="计时迭代数")
    p.add_argument("--warmup", type=int, default=8)
    p.add_argument("--lr", type=float, default=0.0,
                   help="0 = 按 linear rule 自动推（基线 bs8 -> 1e-4）")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--opts", nargs="*", default=[], help="额外 cfg 覆盖（透传给 detectron2）")
    return p.parse_args()


def main():
    args = parse_args()
    torch.cuda.set_device(args.gpu)

    from train_net import Trainer, setup  # noqa

    class _Args:
        config_file = args.config_file
        eval_only = False
        num_gpus = 1
        num_machines = 1
        machine_rank = 0
        dist_url = "tcp://127.0.0.1:0"
        opts = []

    opts = ["SOLVER.IMS_PER_BATCH", str(args.bs)]
    lr = args.lr if args.lr > 0 else 1e-4 * args.bs / 8.0   # linear rule（基线 bs8 = 1e-4）
    opts += ["SOLVER.BASE_LR", str(lr)]
    if args.no_amp:
        opts += ["SOLVER.AMP.ENABLED", "False"]
    opts += list(args.opts)

    a = _Args()
    a.opts = opts
    cfg = setup(a)
    from diffusiondet.data_register import register_all
    register_all(cfg=cfg, verbose=False)

    model = Trainer.build_model(cfg)
    model.train()
    opt = Trainer.build_optimizer(cfg, model)
    loader = Trainer.build_train_loader(cfg)

    torch.cuda.reset_peak_memory_stats()
    amp = cfg.SOLVER.AMP.ENABLED
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    it = iter(loader)
    times = []
    for step in range(args.warmup + args.iters):
        try:
            data = next(it)
        except StopIteration:
            it = iter(loader)
            data = next(it)
        t0 = time.perf_counter()
        with torch.cuda.amp.autocast(enabled=amp):
            loss_dict = model(data)
            loss = sum(v for k, v in loss_dict.items() if "loss" in k)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        if step >= args.warmup:
            times.append(dt)

    peak = torch.cuda.max_memory_allocated() / 1024 ** 2
    med = sorted(times)[len(times) // 2]
    print("\n{:<40} {:>8} {:>9} {:>9} {:>9} {:>10}".format(
        "config", "s/iter", "peak_MiB", "samp/s", "ms/samp", "lr"))
    print("{:<40} {:>8.3f} {:>9.0f} {:>9.1f} {:>9.1f} {:>10.2e}".format(
        "bs={} fp16={}".format(args.bs, amp),
        med, peak, args.bs / med, 1000.0 * med / args.bs, lr))


if __name__ == "__main__":
    main()
