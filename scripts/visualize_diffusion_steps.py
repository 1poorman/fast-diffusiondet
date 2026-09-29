# -*- coding: utf-8 -*-
# 扩散模块（去噪过程）逐步可视化 —— 用 SDD 的真实图像
# ============================================================================
# 画什么：
#   把 DiffusionDet 推理时的反向扩散（去噪）过程按"每次 head forward（NFE）"
#   展开成一张接触表（contact sheet）：
#     * 行 A "input x_t"：该步喂给 head 的当前噪声框（300 个 proposal，全部画出）
#     * 行 B "pred x0"：该步模型直接给出的去噪预测 x0_hat（= 可直接解读的检测框，
#       按 score 过滤上色）
#   列 0 是参照系：上行画 GT 框，下行画最终检测框 + GT 对照。
#   于是整张图就是「纯噪声框 -> 逐步去噪 -> 最终检测」的完整步骤序列。
#
# 为什么这样实现（与仓库既有契约对齐）：
#   1. 前处理 / backbone 特征提取与 detector.forward 的推理分支逐行一致
#      （diffusiondet/detector.py:408-417）；
#   2. 采样走官方入口 diffusiondet.solvers.run_sampler，不自己重写去噪循环，
#      保证与评测口径（BOX_RENEWAL/USE_ENSEMBLE=False，FEP 主矩阵）一致；
#   3. 逐步快照的做法是给 model.make_denoise_fn 返回的 DenoiseFn 包一层：
#      所有 solver（ddim / heun / euler / dpm_v3 / edm_*）都只会通过
#      DenoiseFn.call_model_predictions 调 head，因此包一层即可对全部 solver
#      通用地记录每步的 (x_t, x0_hat, class logits)，记录的步数就是真实 NFE。
#
# 运行（示例，环境见 docs/BLUEPRINT.md）：
#   FASTDD_DATASET=$PWD/datasets CUDA_VISIBLE_DEVICES=1 \
#   ~/.conda/envs/fastdiff/bin/python scripts/visualize_diffusion_steps.py \
#       --config-file configs/lab/sdd.res50.bs8.yaml \
#       --ckpt output/lab/sdd.res50.bs8/model_final.pth \
#       --solver ddim --steps 10 --out results/figures/vis
#
# 产出：results/figures/vis/sdd_diffusion_steps_<solver>_nfe<nfe>.jpg
import argparse
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_net import Trainer, setup  # noqa: E402

# 与 visualize_compare.py 一致的画框规则
DEFAULT_SCORE_THRESH = 0.5
# 面板内文字统一用 ASCII（cv2.putText 不渲染中文；中文说明走 stdout）
TXT_LINE = 26


def parse_args():
    p = argparse.ArgumentParser(
        description="逐步可视化 DiffusionDet 的去噪过程（SDD 图像）")
    p.add_argument("--config-file", default="configs/lab/sdd.res50.bs8.yaml")
    p.add_argument("--ckpt", required=True, help="训练得到的 checkpoint（model_final.pth）")
    p.add_argument("--solver", default=None,
                   help="ddim | euler | heun | dpm_v3 | edm_euler | edm_heun（默认取 cfg）")
    p.add_argument("--steps", type=int, default=10, help="NFE 预算（cfg.SAMPLE_STEP）")
    p.add_argument("--dataset", default=None, help="取图用的数据集名，默认 cfg.DATASETS.TEST[0]")
    p.add_argument("--image", default=None, help="直接指定图像路径（默认取 val 集第一张有 GT 的图）")
    p.add_argument("--image-index", type=int, default=0, help="在候选图中的序号")
    p.add_argument("--seed", type=int, default=42, help="初始噪声 seed（同 seed 可复现）")
    p.add_argument("--score-thresh", type=float, default=DEFAULT_SCORE_THRESH)
    p.add_argument("--max-pred", type=int, default=60, help="预测面板最多画多少个框（防糊图）")
    p.add_argument("--max-noise-draw", type=int, default=300, help="噪声面板最多画多少个 proposal 框")
    p.add_argument("--panel-h", type=int, default=300, help="接触表中每个面板的高度(px)")
    p.add_argument("--cols-per-block", type=int, default=6, help="每行放几列，超出则换块竖排")
    p.add_argument("--out", default="results/figures/vis")
    p.add_argument("--name", default=None, help="输出文件名（默认按 solver/NFE 自动命名）")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--opts", nargs="*", default=[], help="额外的 detectron2 cfg 覆盖")
    return p.parse_args()


# --------------------------------------------------------------------------
# 坐标变换：norm 域 (cx,cy,w,h) ∈ [-scale,scale]  <->  图像绝对 xyxy
# --------------------------------------------------------------------------
def norm_to_abs_xyxy(norm, whwh, scale):
    """与 detector.model_predictions 的输入变换完全一致（含 clamp）。

    norm: (B, P, 4) / whwh: (B, 4) = [w, h, w, h]；返回 (B, P, 4) 绝对 xyxy。
    """
    from diffusiondet.util.box_ops import box_cxcywh_to_xyxy

    x = torch.clamp(norm, min=-1.0 * scale, max=scale)
    c = ((x / scale) + 1.0) / 2.0
    c = torch.cat([c[..., :2], c[..., 2:].clamp(min=1e-4)], dim=-1)
    return box_cxcywh_to_xyxy(c) * whwh[:, None, :]


# --------------------------------------------------------------------------
# 画图工具（全程在 RGB 数组上画，最后统一 [:, :, ::-1] 交给 cv2.imwrite）
# --------------------------------------------------------------------------
def put_text(img, text, org, scale=0.5, color=(255, 255, 255), thickness=1):
    """带黑色描边的文字，保证在任何底色上都可读。"""
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0),
                thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color,
                thickness, cv2.LINE_AA)


def make_band(width, text, band_h=TXT_LINE, color=(255, 255, 255)):
    band = np.full((band_h, width, 3), 255, np.uint8)
    put_text(band, text, (8, band_h - 8), scale=0.5, color=(0, 0, 0))
    return band


def clip_xyxy(boxes, w, h):
    b = boxes.copy() if isinstance(boxes, np.ndarray) else np.asarray(boxes)
    b[:, 0] = np.clip(b[:, 0], 0, w - 1)
    b[:, 1] = np.clip(b[:, 1], 0, h - 1)
    b[:, 2] = np.clip(b[:, 2], 0, w - 1)
    b[:, 3] = np.clip(b[:, 3], 0, h - 1)
    return b


def draw_caption_bar(panel, caption, bottom=None):
    """在面板顶部（或底部）压一条半透明白条再写字，保证任何底色都清晰。"""
    if not caption:
        return panel
    h, w = panel.shape[:2]
    bh = TXT_LINE
    y0 = 0 if bottom is None else h - bh
    bar = panel[y0:y0 + bh].copy()
    white = np.full_like(bar, 255)
    panel[y0:y0 + bh] = cv2.addWeighted(bar, 0.25, white, 0.75, 0)
    put_text(panel, caption, (6, y0 + bh - 8), scale=0.5, color=(20, 20, 20))
    return panel


def draw_noise_panel(image_rgb, boxes, caption=None, max_draw=200):
    """行 A：把当前噪声框 x_t 叠在图上（细灰线 + 半透明，不遮挡图像本身）。"""
    panel = image_rgb.copy()
    h, w = panel.shape[:2]
    if len(boxes):
        boxes = clip_xyxy(boxes[:max_draw], w, h)
        layer = panel.copy()
        for x1, y1, x2, y2 in boxes:
            cv2.rectangle(layer, (int(x1), int(y1)), (int(x2), int(y2)),
                          (70, 70, 70), 1)
        panel = cv2.addWeighted(panel, 0.5, layer, 0.5, 0)
    return draw_caption_bar(panel, caption)


def draw_pred_panel(image_rgb, boxes, scores, labels, meta, caption=None, extra=None):
    """行 B：用 detectron2 Visualizer 画去噪预测框（与仓库其它图同款画风）。"""
    from detectron2.structures import Boxes, Instances
    from detectron2.utils.visualizer import Visualizer

    panel = image_rgb.copy()
    h, w = panel.shape[:2]
    inst = Instances((h, w))
    if len(boxes):
        inst.pred_boxes = Boxes(torch.as_tensor(clip_xyxy(boxes, w, h), dtype=torch.float32))
        inst.scores = torch.as_tensor(scores, dtype=torch.float32)
        inst.pred_classes = torch.as_tensor(labels, dtype=torch.long)
    else:
        inst.pred_boxes = Boxes(torch.zeros((0, 4)))
        inst.scores = torch.zeros((0,))
        inst.pred_classes = torch.zeros((0,), dtype=torch.long)
    v = Visualizer(panel, meta, scale=1.0)
    panel = v.draw_instance_predictions(inst).get_image()
    panel = draw_caption_bar(panel, caption)
    if extra:
        panel = draw_caption_bar(panel, extra, bottom=True)
    return panel


def draw_gt_panel(image_rgb, gt_xyxy, caption=None):
    panel = image_rgb.copy()
    for x1, y1, x2, y2 in clip_xyxy(gt_xyxy, panel.shape[1], panel.shape[0]):
        cv2.rectangle(panel, (int(x1), int(y1)), (int(x2), int(y2)), (0, 160, 0), 2)
    return draw_caption_bar(panel, caption)


def fit_height(panel, target_h):
    h, w = panel.shape[:2]
    return cv2.resize(panel, (max(1, int(round(w * target_h / h))), target_h),
                      interpolation=cv2.INTER_AREA)


# --------------------------------------------------------------------------
def resolve_meta(cfg, num_classes):
    """构造 Visualizer 用的 metadata（只依赖 thing_classes.get）。

    类别数与模型不符时（ckpt 与 config 不匹配）退回占位类名：
    Visualizer._create_text_labels 直接按 label 索引 thing_classes，
    长度不符会 IndexError，所以这里必须对齐到 num_classes。
    注意不能改 MetadataCatalog（detectron2 的 Metadata 禁止改已注册值）。
    """
    from detectron2.data import DatasetCatalog, MetadataCatalog

    name = cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused__"
    tc = []
    if name in DatasetCatalog.list():
        tc = list(getattr(MetadataCatalog.get(name), "thing_classes", None) or [])
    if len(tc) != num_classes:
        if len(tc):
            print(f"[vis] 警告：{name} 类别数({len(tc)}) != 模型类别数({num_classes})，"
                  f"改用占位类名（ckpt 与 config 不匹配？）")
        tc = [f"class_{i}" for i in range(num_classes)]
    return {"thing_classes": tc}


def pick_image(args, cfg):
    """返回 (file_name, gt_anns, dataset_name)。--image 时尝试回查 GT。"""
    from detectron2.data import DatasetCatalog

    ds = args.dataset or (cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else None)
    dicts = []
    if ds and ds in DatasetCatalog.list():
        dicts = DatasetCatalog.get(ds)

    if args.image:
        fname = os.path.abspath(args.image)
        gt = []
        for d in dicts:
            if os.path.abspath(d["file_name"]) == fname:
                gt = d.get("annotations", [])
                break
        return fname, gt, ds

    if not dicts:
        raise FileNotFoundError(
            f"数据集 {ds} 未注册，无法自动取图；请用 --image 指定图像，"
            f"或设置 FASTDD_DATASET 指向数据集根目录。")
    cand = [d for d in dicts if len(d.get("annotations", [])) > 0]
    if not cand:
        raise RuntimeError(f"{ds} 中没有带标注的图像。")
    d = cand[min(args.image_index, len(cand) - 1)]
    return d["file_name"], d.get("annotations", []), ds


def main():
    args = parse_args()
    torch.cuda.set_device(args.gpu)

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
    cfg.MODEL.DEVICE = f"cuda:{args.gpu}"
    # FEP 主矩阵口径：renewal / ensemble 一律关闭（多步对比才公平）
    cfg.MODEL.DiffusionDet.BOX_RENEWAL = False
    cfg.MODEL.DiffusionDet.USE_ENSEMBLE = False
    cfg.MODEL.DiffusionDet.SAMPLE_STEP = args.steps
    if args.solver:
        cfg.MODEL.DiffusionDet.SOLVER = args.solver
    if args.dataset:
        cfg.DATASETS.TEST = (args.dataset,)
    # EDM 范式必须配 σ 空间求解器，否则 VP 的 DDIM 更新公式与 VE 模型不自洽
    if cfg.MODEL.DiffusionDet.FORMULATION == "edm" and \
            not cfg.MODEL.DiffusionDet.SOLVER.startswith("edm_"):
        print(f"[vis] FORMULATION=edm 但 solver={cfg.MODEL.DiffusionDet.SOLVER}，"
              f"自动切到 edm_heun")
        cfg.MODEL.DiffusionDet.SOLVER = "edm_heun"
    if args.opts:
        cfg.merge_from_list(args.opts)
    cfg.freeze()

    from diffusiondet.data_register import register_all
    register_all(cfg=cfg, verbose=False)

    model = Trainer.build_model(cfg)
    from detectron2.checkpoint import DetectionCheckpointer
    DetectionCheckpointer(model, save_dir="").resume_or_load(cfg.MODEL.WEIGHTS, resume=False)
    model.eval()

    meta = resolve_meta(cfg, model.num_classes)
    fname, gt_anns, ds = pick_image(args, cfg)
    print(f"[vis] image = {fname}  (dataset={ds}, gt={len(gt_anns)})")
    print(f"[vis] solver = {model.solver_name}  NFE预算 = {args.steps}  "
          f"formulation = {model.formulation}  scale = {model.scale}")

    # ---------------- 读图 + 与 detector.forward 一致的前处理 ----------------
    img_bgr = cv2.imread(fname)
    if img_bgr is None:
        raise FileNotFoundError(f"读不到图像：{fname}")
    h0, w0 = img_bgr.shape[:2]
    from detectron2.data import transforms as T
    aug = T.ResizeShortestEdge([cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MAX_SIZE_TEST],
                              cfg.INPUT.MAX_SIZE_TEST)
    tfm = aug.get_transform(img_bgr)
    image = tfm.apply_image(img_bgr)            # BGR，resize 后
    image_rgb = image[:, :, ::-1].copy()        # 后续统一在 RGB 上画
    image_t = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
    batched_inputs = [{"image": image_t, "height": h0, "width": w0}]

    # GT 变换到 resize 后的坐标系
    gt_xyxy = np.zeros((0, 4), np.float32)
    if gt_anns:
        raw = np.array([[a["bbox"][0], a["bbox"][1],
                         a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]]
                        for a in gt_anns], dtype=np.float32)
        gt_xyxy = tfm.apply_box(raw).astype(np.float32)

    # ---------------- 特征提取 + 逐步记录去噪状态 ----------------
    snapshots = []
    with torch.no_grad():
        images, images_whwh = model.preprocess_image(batched_inputs)
        src = model.backbone(images.tensor)
        feats = [src[f] for f in model.in_features]

        batch, shape = 1, (1, model.num_proposals, 4)
        torch.manual_seed(args.seed)
        init_noise = torch.randn(shape, device=model.device)
        if model.formulation == "edm":      # 与 edm.py 默认初值同尺度
            init_noise = init_noise * model.edm_sigma_max

        from diffusiondet.solvers import run_sampler

        orig_make = model.make_denoise_fn

        def patched_make(backbone_feats, whwh):
            """包一层 DenoiseFn：所有 solver 调 head 都必经此处 -> 逐步快照。"""
            fn = orig_make(backbone_feats, whwh)
            orig_cmp = fn.call_model_predictions

            def cmp(x, t):
                preds, oc, oco = orig_cmp(x, t)
                snapshots.append({
                    "x_in": x.detach().clone(),                    # 当前噪声框 x_t
                    "x_start": preds.pred_x_start.detach().clone(),  # x0_hat（norm 域）
                    "logits": oc[-1].detach().clone(),             # 类别 logits
                    "coord": oco[-1].detach().clone(),             # x0_hat（绝对 xyxy）
                    "t": float(t.reshape(-1)[0].item()),
                })
                return preds, oc, oco

            fn.call_model_predictions = cmp
            return fn

        model.make_denoise_fn = patched_make
        try:
            results = run_sampler(model, batched_inputs, feats, images_whwh, images,
                                  do_postprocess=True, noise=init_noise)
        finally:
            model.make_denoise_fn = orig_make

    n_steps = len(snapshots)
    if n_steps == 0:
        raise RuntimeError("没有记录到任何去噪步，采样器未调用 denoise_fn？")

    # ---------------- 逐步出框点数 + 统计 ----------------
    def step_boxes(snap, thresh):
        logits = snap["logits"][0]                     # (P, C)
        if model.use_focal:
            prob = logits.sigmoid()
            scores, labels = prob.max(dim=-1)
        else:
            scores, labels = logits.softmax(-1)[..., :-1].max(-1)
        keep = scores > thresh
        b = snap["coord"][0][keep].cpu().numpy()
        s = scores[keep].cpu().numpy()
        l = labels[keep].cpu().numpy()
        if len(s) > args.max_pred:                     # 只保留最靠前的一批，防糊
            order = np.argsort(-s)[: args.max_pred]
            b, s, l = b[order], s[order], l[order]
        return b, s, l, float(scores.max().item())

    print(f"\n[vis] 去噪步骤（solver={model.solver_name}, 真实 NFE={n_steps}, "
          f"score>{args.score_thresh}）")
    print(f"{'#':>3} {'t/sigma':>9} {'n_box':>6} {'max_score':>10}")

    # ---------------- 拼图 ----------------
    # 列 0 = 参照（上行 GT，下行最终检测+GT），列 1..N = 各去噪步
    # 面板内容先在原始（网络输入）分辨率下画，缩放后再写说明文字，否则字会被缩小。
    colA, colB, capA, capB, extraB = [], [], [], [], []

    colA.append(draw_gt_panel(image_rgb, gt_xyxy, None) if len(gt_xyxy)
                else draw_gt_panel(image_rgb, np.zeros((0, 4)), None))
    capA.append("GT boxes" if len(gt_xyxy) else "GT (none)")

    inst_final = results[0]["instances"]
    fin = inst_final[inst_final.scores > args.score_thresh]
    fb = fin.pred_boxes.tensor.cpu().numpy()
    fs = fin.scores.cpu().numpy()
    fl = fin.pred_classes.cpu().numpy()
    if len(fs) > args.max_pred:
        order = np.argsort(-fs)[: args.max_pred]
        fb, fs, fl = fb[order], fs[order], fl[order]
    panel = draw_pred_panel(image_rgb, fb, fs, fl, meta)
    if len(gt_xyxy):   # 最终检测面板上再叠 GT（绿框）做对照
        for x1, y1, x2, y2 in clip_xyxy(gt_xyxy, image_rgb.shape[1], image_rgb.shape[0]):
            cv2.rectangle(panel, (int(x1), int(y1)), (int(x2), int(y2)), (0, 160, 0), 2)
    colB.append(panel)
    capB.append(f"final  n={len(fs)}")
    extraB.append(f"NFE={n_steps}  green=GT")

    for k, snap in enumerate(snapshots):
        t_lab = f"t={snap['t']:.0f}" if model.formulation != "edm" else f"sigma={snap['t']:.2f}"
        xb = norm_to_abs_xyxy(snap["x_in"], images_whwh, model.scale)[0].cpu().numpy()
        pb, ps, pl, mx = step_boxes(snap, args.score_thresh)
        print(f"{k + 1:>3} {snap['t']:>9.2f} {len(ps):>6} {mx:>10.3f}")
        colA.append(draw_noise_panel(image_rgb, xb, None, args.max_noise_draw))
        capA.append(f"x_t  #{k + 1}/{n_steps}  {t_lab}")
        colB.append(draw_pred_panel(image_rgb, pb, ps, pl, meta))
        capB.append(f"x0  #{k + 1}/{n_steps}  n={len(ps)}")
        extraB.append(f"{t_lab}  max={mx:.2f}")

    # 面板统一高度 -> 说明文字 -> 按块竖排
    A = [draw_caption_bar(fit_height(p, args.panel_h), c) for p, c in zip(colA, capA)]
    B = []
    for p, c, e in zip(colB, capB, extraB):
        p = draw_caption_bar(fit_height(p, args.panel_h), c)
        B.append(draw_caption_bar(p, e, bottom=True))

    cols = len(A)
    nblk = max(1, -(-cols // max(1, args.cols_per_block)))
    step = max(1, -(-cols // nblk))   # 均衡分块，避免最后一块只放一列

    blocks = []
    for s in range(0, cols, step):
        rowA = np.concatenate(A[s:s + step], axis=1)
        rowB = np.concatenate(B[s:s + step], axis=1)
        wmax = max(rowA.shape[1], rowB.shape[1])
        pad = lambda r: np.pad(r, ((0, 0), (0, wmax - r.shape[1]), (0, 0)),
                               constant_values=255)
        rowA, rowB = pad(rowA), pad(rowB)
        gap = np.full((4, wmax, 3), 255, np.uint8)
        blocks.append(np.concatenate([rowA, gap, rowB], axis=0))

    width = max(b.shape[1] for b in blocks)
    blocks = [np.pad(b, ((0, 0), (0, width - b.shape[1]), (0, 0)), constant_values=255)
              for b in blocks]
    sep = np.full((10, width, 3), 235, np.uint8)

    head = make_band(
        width,
        f"DiffusionDet denoising steps | {os.path.basename(args.config_file)} | "
        f"solver={model.solver_name} NFE={n_steps} | img={os.path.basename(fname)} "
        f"| seed={args.seed}", band_h=30)
    foot = make_band(
        width,
        "row A: noisy proposals x_t fed to head (all, clamped)   "
        "row B: model x0 prediction (colored, score>%.2f)   "
        "col 0: GT (green) / final detection" % args.score_thresh, band_h=30)

    body = []
    for i, b in enumerate(blocks):
        if i:
            body.append(sep)
        body.append(b)
    fig = np.concatenate([head] + body + [foot], axis=0)

    os.makedirs(args.out, exist_ok=True)
    name = args.name or f"sdd_diffusion_steps_{model.solver_name}_nfe{n_steps}.jpg"
    out_path = os.path.join(args.out, name)
    cv2.imwrite(out_path, fig[:, :, ::-1])
    print(f"\n[vis] saved -> {out_path}  ({fig.shape[1]}x{fig.shape[0]})")


if __name__ == "__main__":
    main()
