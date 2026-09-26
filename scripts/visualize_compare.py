# 可视化对比：检测效果图（多 solver×NFE）+ Grad-CAM++ 热力图
# 用法：python scripts/visualize_compare.py --ckpt <pth> [--config cfg] [--num-images 4]
# 产出：results/figures/vis/grid_det.jpg（检测对比横条）、grid_cam.jpg（热力图横条）
# 每个配置一行（标题条 + 所有图横向拼接），配置纵向堆叠，行内同图同 seed 可比。
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_net import Trainer, setup  # noqa: E402

CONFIGS = [("ddim", 1), ("ddim", 2), ("dpm_v3", 2), ("heun", 3)]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config-file", default="configs/lab/sdd.res50.bs8.yaml")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--dataset", default=None)
    p.add_argument("--num-images", type=int, default=4)
    p.add_argument("--out", default="results/figures/vis")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--score-thresh", type=float, default=0.5)
    return p.parse_args()


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
    if args.dataset:
        cfg.DATASETS.TEST = (args.dataset,)
    cfg.freeze()
    from diffusiondet.data_register import register_all
    register_all(cfg=cfg)

    model = Trainer.build_model(cfg)
    from detectron2.checkpoint import DetectionCheckpointer
    DetectionCheckpointer(model).resume_or_load(cfg.MODEL.WEIGHTS, resume=False)
    model.eval()

    from detectron2.data import DatasetCatalog, MetadataCatalog
    dicts = DatasetCatalog.get(cfg.DATASETS.TEST[0])
    meta = MetadataCatalog.get(cfg.DATASETS.TEST[0])
    dicts = [d for d in dicts if len(d.get("annotations", [])) > 0][: args.num_images]

    shapes = {k: s.stride for k, s in model.backbone.output_shape().items()}
    # CAM 层从 head 实际使用的 in_features 里选 stride 最大的（p6 等
    # 未进 head 的层在 make_denoise_fn 的 feats 列表里不存在）
    in_feats = model.in_features
    last_key = max(in_feats, key=lambda k: shapes[k])
    store = {}

    def fwd_hook(m, i, o):
        store["act"] = o.detach()

    def bwd_hook(m, gi, go):
        store["grad"] = go[0].detach()

    # FPN 的 p5 是动态输出（无同名子模块），且模块级 full_backward_hook 对
    # dict 输出不触发——改为 forward hook 里对 p5 张量直接 register_hook
    # （backward 经过该张量必然触发，比模块级 hook 可靠）
    def fwd_hook(m, i, o):
        p5 = o[last_key]
        store["act"] = p5.detach()
        # no_grad 下的推理前向无 grad_fn（检测面板用）；只有 CAM 的带梯度
        # 前向才注册 tensor hook
        if p5.requires_grad:
            p5.register_hook(lambda g: store.__setitem__("grad", g.detach()))

    model.backbone.register_forward_hook(fwd_hook)
    print(f"[vis] CAM layer={last_key} stride={shapes[last_key]}", flush=True)

    from detectron2.data import transforms as T
    from detectron2.utils.visualizer import Visualizer
    from diffusiondet.util.box_ops import box_cxcywh_to_xyxy
    aug = T.ResizeShortestEdge([cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MAX_SIZE_TEST],
                               cfg.INPUT.MAX_SIZE_TEST)

    det_panels = {c: [] for c in CONFIGS}
    cam_panels = {c: [] for c in CONFIGS}

    for k, d in enumerate(dicts):
        img_bgr = cv2.imread(d["file_name"])
        h, w = img_bgr.shape[:2]
        image = aug.get_transform(img_bgr).apply_image(img_bgr)
        image_t = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        batched_inputs = [{"image": image_t, "height": h, "width": w}]

        for (solver, steps) in CONFIGS:
            model.solver_name = solver
            model.sampling_timesteps = steps
            model.box_renewal = False
            model.use_ensemble = False
            torch.manual_seed(42)

            # ---- 检测面板：完整推理 ----
            with torch.no_grad():
                results = model(batched_inputs)
            inst = results[0]["instances"]
            keep = inst.scores > args.score_thresh
            v = Visualizer(image[:, :, ::-1].copy(), meta, scale=1.0)
            out = v.draw_instance_predictions(inst[keep].to("cpu"))
            det_panels[(solver, steps)].append(out.get_image())

            # ---- Grad-CAM++：重放采样到 x_start，head logits 之和为目标 ----
            with torch.no_grad():
                images, images_whwh = model.preprocess_image(batched_inputs)
            src_g = model.backbone(images.tensor)
            feats = [src_g[f] for f in model.in_features]
            denoise_fn = model.make_denoise_fn(feats, images_whwh)
            if solver == "ddim":
                # DDIM 的 do_postprocess=False 返回 Instances 列表（原契约），
                # CAM 需要张量 -> 手动重放 eta=0 确定性更新（CAM 定性可视化足够）
                from diffusiondet.solvers.schedule import build_time_pairs
                dn = model.make_denoise_fn(feats, images_whwh)
                pairs, _ = build_time_pairs(model.num_timesteps, steps)
                x = torch.randn((1, model.num_proposals, 4), device=device)
                ac = model.alphas_cumprod
                with torch.no_grad():
                    for t, tn in pairs:
                        tc = torch.full((1,), t, device=device, dtype=torch.long)
                        _, xs = dn(x, tc)
                        if tn < 0:
                            x = xs
                            break
                        a, an = ac[t].sqrt(), ac[tn].sqrt()
                        x = xs * an.sqrt() + (1 - an).sqrt() * \
                            ((x - xs * a.sqrt()) / (1 - ac[t]).sqrt())
                x_final = x
            else:
                with torch.no_grad():
                    from diffusiondet.solvers import run_sampler
                    x_final, _, _ = run_sampler(
                        model, batched_inputs, feats, images_whwh, images,
                        do_postprocess=False)
            x_in = torch.clamp(x_final.detach(), min=-1 * model.scale, max=model.scale)
            x_boxes = box_cxcywh_to_xyxy(((x_in / model.scale) + 1) / 2.) * images_whwh[:, None, :]
            outputs_class, _ = model.head(feats, x_boxes,
                                          torch.zeros(1, device=device), None)
            target = outputs_class[-1].sum()
            model.zero_grad(set_to_none=True)
            target.backward()

            A, G = store["act"][0], store["grad"][0]
            anum = G.pow(2)
            aden = 2 * anum + (A * G.pow(3)).sum(dim=(1, 2), keepdim=True)
            alpha = anum / (aden + 1e-8)
            w = (alpha * F.relu(G)).sum(dim=(1, 2))
            cam = F.relu((w[:, None, None] * A).sum(dim=0))
            cam = (cam - cam.min()) / (cam.max() + 1e-8)
            cam = cv2.resize(cam.cpu().numpy(), (image.shape[1], image.shape[0]))
            heat = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
            cam_panels[(solver, steps)].append(
                cv2.addWeighted(image, 0.55, heat, 0.45, 0)[:, :, ::-1])
            print(f"[vis] img{k} {solver}@{steps}: dets={int(keep.sum())} "
                  f"target={target.item():.1f}", flush=True)
            model.zero_grad(set_to_none=True)

    # ---- 拼接：每配置一行（标题条 + 各图横排），宽度不齐处白底补齐 ----
    os.makedirs(args.out, exist_ok=True)
    for name, panels in [("grid_det", det_panels), ("grid_cam", cam_panels)]:
        rows = []
        for c in CONFIGS:
            ps = panels[c]
            hmax = max(p.shape[0] for p in ps)
            wsum = sum(p.shape[1] for p in ps)
            band = np.full((30, wsum, 3), 255, np.uint8)
            cv2.putText(band, f"{c[0]} steps={c[1]}", (8, 21),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
            ps = [np.pad(p, ((0, hmax - p.shape[0]), (0, 0), (0, 0)),
                         constant_values=255) for p in ps]
            strip = np.concatenate(ps, axis=1)  # 各图横排
            rows.append(np.concatenate([band, strip], axis=0))  # 标题条在上
        wmax = max(r.shape[1] for r in rows)
        rows = [np.pad(r, ((0, 0), (0, wmax - r.shape[1]), (0, 0)),
                       constant_values=255) for r in rows]
        cv2.imwrite(f"{args.out}/{name}.jpg", np.concatenate(rows, axis=0)[:, :, ::-1])
        print(f"[vis] saved {args.out}/{name}.jpg")


if __name__ == "__main__":
    main()
