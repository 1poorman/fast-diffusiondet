# -*- coding: utf-8 -*-
# 扩散模块（去噪过程）逐步可视化 —— 用 SDD 的真实图像
# ============================================================================
# 画什么：
#   把 DiffusionDet 推理时的反向扩散（去噪）过程按"每次 head forward（NFE）"
#   展开成一张接触表（contact sheet）：
#     * 行 A "input x_t"：该步喂给 head 的当前噪声框（proposal，白细线）
#     * 行 B "pred x0"：该步模型直接给出的去噪预测框 x0_hat
#       （粗轮廓 + 半透明填充 + 类名/分数标签，可直接解读为检测框）
#   列 0 是参照系：上行画 GT 框，下行画最终检测框 + GT 对照。
#   于是整张图就是「纯噪声框 -> 逐步去噪 -> 最终检测」的完整步骤序列。
#
# 可读性设计（针对"注释不清、框不明显"的反馈）：
#   1. 行 B 不用 detectron2 Visualizer 的 alpha 填充（在图上只剩淡色块），
#      改为「18% 半透明填充 + 3px 实色轮廓 + 实色底白字标签」，轮廓清楚；
#   2. 行 A 的噪声框改画在压暗的图底上、用高对比白色细线（原灰线看不清）；
#   3. 每个面板的说明统一放进一条实白说明条（单行、大字号、无描边避免糊字），
#      行含义由整行标题带 + 底部图例色块说明，不再挤在面板里；
#   4. GT 用绿色粗框 + "GT" 实底标签。
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
import json
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_net import Trainer, setup  # noqa: E402

DEFAULT_SCORE_THRESH = 0.5
CAP_H = 26      # 每个面板顶部的实白说明条高度
ROW_H = 30      # 整行标题带高度
LEG_H = 42      # 底部图例高度
BAND_BG = 238   # 页眉/行标题/图例/补位的统一底色

# 类别配色（BGR/RGB 通用三通道值，在白底说明条与图例上都能看清）
CLASS_COLORS = [
    (60, 76, 231),    # 红
    (219, 152, 52),   # 蓝
    (34, 126, 230),   # 橙
    (182, 89, 155),   # 紫
    (99, 153, 34),    # 绿
    (60, 163, 219),   # 黄
    (182, 89, 155),   # 品红
]
GT_COLOR = (0, 170, 0)          # GT 绿
NOISE_COLOR = (255, 255, 255)   # 噪声框白


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
    p.add_argument("--label-topk", type=int, default=4,
                   help="行 B 最多给几个高分框画 类名+分数 标签（防文字堆叠）")
    p.add_argument("--max-noise-draw", type=int, default=150,
                   help="噪声面板最多画多少个 proposal 框（太多会糊成网格）")
    p.add_argument("--panel-h", type=int, default=320, help="接触表中每个面板的高度(px)")
    p.add_argument("--cols-per-block", type=int, default=6, help="每行放几列，超出则换块竖排")
    p.add_argument("--out", default="results/figures/vis")
    p.add_argument("--name", default=None, help="输出文件名（默认按 solver/NFE 自动命名）")
    p.add_argument("--save-steps-dir", default="",
                   help="把每个 NFE 步骤的检测结果图按全分辨率单独保存到该目录"
                        "（不缩放、不叠说明条，并生成 README 索引）")
    p.add_argument("--save-noise", action="store_true",
                   help="额外保存每步的噪声框面板 x_t（默认只存检测结果 x0）")
    p.add_argument("--save-orig-size", action="store_true",
                   help="结果图另存一份到原始图像分辨率（坐标按 resize 比例还原）")
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
def put_text(img, text, org, scale=0.55, color=(20, 20, 20), thickness=1):
    """单遍写字（不做描边）。字要清楚地写在实底条/实底标签上时用它。"""
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color,
                thickness, cv2.LINE_AA)


def text_w(text, scale=0.55, thickness=1):
    return cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0][0]


def band(width, text, height=CAP_H, bg=(255, 255, 255), fg=(20, 20, 20), scale=0.55):
    """实底文字条（黑/灰字白底，或行标题的反白条）。"""
    b = np.full((height, width, 3), bg, np.uint8)
    put_text(b, text, (8, height - 8), scale=scale, color=fg)
    return b


def clip_xyxy(boxes, w, h):
    b = np.asarray(boxes, dtype=np.float32).copy()
    if not len(b):
        return b.reshape(0, 4)
    b[:, 0] = np.clip(b[:, 0], 0, w - 1)
    b[:, 1] = np.clip(b[:, 1], 0, h - 1)
    b[:, 2] = np.clip(b[:, 2], 0, w - 1)
    b[:, 3] = np.clip(b[:, 3], 0, h - 1)
    return b


def label_bg(img, text, x, y, color, scale=0.45):
    """实色底 + 白字标签（紧贴框上沿，越界时下移/内收）。"""
    tw = text_w(text, scale)
    th = int(round(14 * scale / 0.45)) + 8
    x = int(np.clip(x, 0, max(0, img.shape[1] - tw - 6)))
    y = int(y)
    if y - th < 0:
        y = th + 2                      # 顶部放不下就贴到下面
    cv2.rectangle(img, (x, y - th), (x + tw + 8, y), color, -1)
    put_text(img, text, (x + 4, y - 6), scale=scale, color=(255, 255, 255))


HILITE_COLOR = (236, 84, 44)    # 高亮色（橙红）：该步头部最自信的 proposal


def draw_noise_panel(image_rgb, boxes, scores, max_draw, thick):
    """行 A：当前噪声框 x_t。

    全量框只当"纹理"淡淡铺一层（原 150 条白线会糊成网格、盖住图像），
    再把该步 head 分类分最高的几个 proposal 用橙红粗框高亮出来：
    既不丢"300 个噪声框"的信息，又能直观看清这一步模型在关注哪些框。
    """
    if not len(boxes):
        return image_rgb.copy()
    panel = cv2.addWeighted(image_rgb, 0.72, np.full_like(image_rgb, 30), 0.28, 0)
    h, w = panel.shape[:2]
    b = clip_xyxy(boxes[:max_draw], w, h)

    layer = panel.copy()
    for x1, y1, x2, y2 in b:
        cv2.rectangle(layer, (int(x1), int(y1)), (int(x2), int(y2)), NOISE_COLOR, 1)
    panel = cv2.addWeighted(panel, 0.62, layer, 0.38, 0)

    top = np.argsort(-np.asarray(scores))[:6]
    for i in top:
        x1, y1, x2, y2 = clip_xyxy(boxes[[i]], w, h)[0]
        cv2.rectangle(panel, (int(x1), int(y1)), (int(x2), int(y2)), HILITE_COLOR, thick)
    return panel


def draw_pred_boxes(image_rgb, boxes, scores, labels, thick):
    """行 B：去噪预测框。

    不用 detectron2 Visualizer 的 alpha 填充（在图上只剩淡色块、无轮廓），
    改为「半透明填充标出目标区域 + 粗实色轮廓」，标签在缩放之后再单独写
    （缩放前写会被缩小到看不清，见 add_pred_labels）。
    """
    panel = image_rgb.copy()
    h, w = panel.shape[:2]
    if not len(boxes):
        return panel
    b = clip_xyxy(boxes, w, h)
    order = np.argsort(scores)          # 低分先画，高分在上

    fill = panel.copy()
    for i in order:
        cv2.rectangle(fill, (int(b[i, 0]), int(b[i, 1])), (int(b[i, 2]), int(b[i, 3])),
                      CLASS_COLORS[int(labels[i]) % len(CLASS_COLORS)], -1)
    panel = cv2.addWeighted(fill, 0.22, panel, 0.78, 0)

    for i in order:
        cv2.rectangle(panel, (int(b[i, 0]), int(b[i, 1])), (int(b[i, 2]), int(b[i, 3])),
                      CLASS_COLORS[int(labels[i]) % len(CLASS_COLORS)], thick)
    return panel


def add_pred_labels(panel, boxes, scores, labels, names, ratio, topk=4):
    """在缩放后的面板上写「类名 + 分数」标签（缩放后再写才看得清）。"""
    if not len(boxes) or topk <= 0:
        return panel
    b = np.asarray(boxes) * ratio
    for i in np.argsort(-np.asarray(scores))[:topk]:
        name = names[int(labels[i])] if 0 <= int(labels[i]) < len(names) else str(int(labels[i]))
        label_bg(panel, f"{name} {scores[i]:.2f}", b[i, 0], b[i, 1],
                 CLASS_COLORS[int(labels[i]) % len(CLASS_COLORS)], scale=0.62)
    return panel


def draw_gt_panel(image_rgb, gt_xyxy, thick):
    panel = image_rgb.copy()
    b = clip_xyxy(gt_xyxy, panel.shape[1], panel.shape[0])
    for i, (x1, y1, x2, y2) in enumerate(b):
        cv2.rectangle(panel, (int(x1), int(y1)), (int(x2), int(y2)), GT_COLOR, thick)
    if len(b):
        label_bg(panel, f"GT {len(b)}", b[0, 0], b[0, 1], GT_COLOR, scale=0.7)
    return panel


class ResultSaver:
    """把逐 NFE 步骤的检测结果图**按全分辨率**单独落盘，并生成索引 README。

    为什么单独做这件事：拼图里的面板会被缩到 --panel-h（默认 320px）并叠上说明条，
    只适合阅读；后期编辑/放大看细节/做动画需要的是「一张图一个步骤」的原始尺寸结果。
    这里保存的就是未缩放的绘制结果（模型输入分辨率，或 --save-orig-size 的原始尺寸）。
    """

    def __init__(self, out_dir, header_lines=None):
        self.out_dir = out_dir or ""
        self.rows = []
        self.header = list(header_lines or [])
        if self.out_dir:
            os.makedirs(self.out_dir, exist_ok=True)

    @property
    def enabled(self):
        return bool(self.out_dir)

    def add(self, panel_rgb, filename, info, sub=""):
        """panel_rgb 为 RGB 数组；sub 是可选的子目录（如 ckpt/solver/img）。"""
        if not self.enabled:
            return None
        d = os.path.join(self.out_dir, sub) if sub else self.out_dir
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, filename)
        cv2.imwrite(path, panel_rgb[:, :, ::-1])
        self.rows.append((os.path.relpath(path, self.out_dir), info))
        return path

    @staticmethod
    def _derive(rel):
        """从路径/文件名反推说明，用于补齐早前批次留下的文件（文件名里带了要素）。"""
        parts = rel.replace("\\", "/").split("/")
        base = os.path.splitext(parts[-1])[0]
        where = " / ".join(parts[:-1]) if len(parts) > 1 else "(根目录)"
        size = " [原始尺寸]" if base.endswith("_orig") else ""
        if size:
            base = base[: -len("_orig")]
        if base.startswith("00_gt"):
            return f"{where}：GT 参考"
        if base.startswith("final_"):
            return f"{where}：最终检测（topk+NMS，绿框=GT）{size}"
        toks = base.split("_")
        kv = []
        for t in toks[1:]:
            if t.startswith("n") and t[1:].isdigit():
                kv.append("n=" + t[1:])
            elif t.startswith("max"):
                kv.append("max=" + t[3:])
        step = toks[0].replace("step", "#")
        if "_xt_" in base or any(t.startswith("t") and t[1:].isdigit() for t in toks):
            t_lab = next((t for t in toks if t.startswith("t") and t[1:].isdigit()), "?")
            return f"{where}：{step} 噪声框 x_t（{t_lab.replace('t', 't=')}）{size}"
        return f"{where}：{step} x0 预测 {' '.join(kv)}{size}"

    def close(self):
        if not self.enabled:
            return
        # 累积索引：多次运行（不同求解器/权重）写同一份 README 时不能互相覆盖，
        # 所以把每批的元信息合并进 _index.json，README 由合并结果生成。
        idx_path = os.path.join(self.out_dir, "_index.json")
        merged = {}
        if os.path.exists(idx_path):
            try:
                with open(idx_path, encoding="utf-8") as f:
                    merged = json.load(f)
            except (OSError, ValueError):
                merged = {}
        for rel, info in self.rows:
            merged[rel.replace("\\", "/")] = info

        # 目录里实际存在的图片（含早前批次、未记录元信息的），保证索引完整
        on_disk = set()
        for root, _, files in os.walk(self.out_dir):
            for fn in files:
                if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                    on_disk.add(os.path.relpath(os.path.join(root, fn),
                                                self.out_dir).replace("\\", "/"))
        for rel in on_disk:
            merged.setdefault(rel, self._derive(rel))
        with open(idx_path, "w", encoding="utf-8") as f:
            json.dump(merged, f, ensure_ascii=False, indent=2, sort_keys=True)

        n_batch = len(self.rows)
        lines = ["# 检测结果图（逐 NFE 步骤，全分辨率）", ""] + self.header
        lines += ["", f"本批新增 {n_batch} 张；目录内合计 {len(on_disk)} 张。", "",
                  "| 文件 | 说明 |", "|---|---|"]
        lines += [f"| `{f}` | {merged.get(f, self._derive(f))} |"
                  for f in sorted(on_disk)]
        if len(on_disk) != len(merged):
            lines += ["", f"> 注：索引里另有 {len(merged) - len(on_disk)} 条记录对应的文件"
                          f"已不在目录中（可能被手工删除）。"]
        lines += [
            "", "说明：", "",
            "- 文件是全分辨率的检测结果图（未经拼图缩放），可直接用于放大、动画或二次编辑；",
            "- `n=` 是过阈值（score > thr）的检出数，`max=` 是全部 proposal 的最高分；",
            "- 逐步骤的 `x0` 是原始 top-k 预测、**未做 NMS**，中间步骤常有重叠框；",
            "  带 NMS 的结果单列为 `final_topkNMS_*.jpg`；",
            "- 机器可读索引见同目录 `_index.json`；",
            "- 拼图总览见 `results/figures/vis/`，原图与标注见 `datasets/vis_images/`。",
        ]
        with open(os.path.join(self.out_dir, "README.md"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")


def save_pred_pair(saver, filename, info, work_rgb, boxes, scores, labels, names,
                   topk, save_thick, sub="", orig_rgb=None, orig_thick=None,
                   orig_scale=1.0, save_orig=False, gt_xyxy=None):
    """把一张检测结果图按**全分辨率**落盘（不缩放、不叠说明条）。

    save_orig=True 时再用原始图像尺寸渲染一份（坐标乘 orig_scale 还原）。
    两个可视化脚本共用，避免各写一份落盘逻辑导致文件名/画法不一致。
    """
    if not saver.enabled:
        return
    panel = draw_pred_boxes(work_rgb, boxes, scores, labels, save_thick)
    if gt_xyxy is not None and len(gt_xyxy):
        for x1, y1, x2, y2 in clip_xyxy(gt_xyxy, work_rgb.shape[1], work_rgb.shape[0]):
            cv2.rectangle(panel, (int(x1), int(y1)), (int(x2), int(y2)),
                          GT_COLOR, max(2, save_thick - 1))
    panel = add_pred_labels(panel, boxes, scores, labels, names, 1.0, topk)
    saver.add(panel, filename, info, sub=sub)

    if save_orig and orig_rgb is not None:
        ob = np.asarray(boxes) * orig_scale
        p = draw_pred_boxes(orig_rgb, ob, scores, labels, orig_thick)
        if gt_xyxy is not None and len(gt_xyxy):
            g = np.asarray(gt_xyxy) * orig_scale
            for x1, y1, x2, y2 in clip_xyxy(g, orig_rgb.shape[1], orig_rgb.shape[0]):
                cv2.rectangle(p, (int(x1), int(y1)), (int(x2), int(y2)),
                              GT_COLOR, max(2, orig_thick - 1))
        p = add_pred_labels(p, ob, scores, labels, names, 1.0, topk)
        saver.add(p, filename.replace(".jpg", "_orig.jpg"), info + " [原始尺寸]", sub=sub)


def run_with_snapshots(model, batched_inputs, seed=42):
    """跑一次官方采样入口，并记录每一步的 (x_t, logits, x0_hat)。

    返回 (results, snapshots, images_whwh)。所有 solver 都经
    DenoiseFn.call_model_predictions 调 head，所以包一层即可通用记录，
    记录的条数就是真实 NFE（heun 为 2k-1、dpm_v3 含 corrector 调用）。

    供 visualize_diffusion_steps / visualize_diffusion_matrix 共用，
    避免两处各写一份采样逻辑造成口径漂移。
    """
    snapshots = []
    with torch.no_grad():
        images, images_whwh = model.preprocess_image(batched_inputs)
        src = model.backbone(images.tensor)
        feats = [src[f] for f in model.in_features]

        torch.manual_seed(seed)
        init_noise = torch.randn((1, model.num_proposals, 4), device=model.device)
        if model.formulation == "edm":      # 与 edm.py 默认初值同尺度
            init_noise = init_noise * model.edm_sigma_max

        from diffusiondet.solvers import run_sampler

        orig_make = model.make_denoise_fn

        def patched_make(backbone_feats, whwh):
            fn = orig_make(backbone_feats, whwh)
            orig_cmp = fn.call_model_predictions

            def cmp(x, t):
                preds, oc, oco = orig_cmp(x, t)
                snapshots.append({
                    "x_in": x.detach().clone(),          # 当前噪声框 x_t
                    "logits": oc[-1].detach().clone(),   # 类别 logits
                    "coord": oco[-1].detach().clone(),   # x0_hat（绝对 xyxy）
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
    return results, snapshots, images_whwh


def step_boxes(model, coord, logits, thresh, max_pred):
    """单步 head 输出的后处理：过阈框 + 每 proposal 的分数。

    coord (1,P,4) 为绝对坐标的 x0_hat，logits (1,P,C)；
    返回 (过阈框, 过阈分数, 过阈类别, 最高分, 全部 proposal 分数)。
    """
    logits = logits[0]                                 # (P, C)
    if model.use_focal:
        scores, labels = logits.sigmoid().max(dim=-1)
    else:
        scores, labels = logits.softmax(-1)[..., :-1].max(-1)
    all_scores = scores.detach().cpu().numpy()
    max_score = float(scores.max().item())
    keep = scores > thresh
    b = coord[0][keep].cpu().numpy()
    s = scores[keep].cpu().numpy()
    l = labels[keep].cpu().numpy()
    if len(s) > max_pred:                              # 只留最靠前的一批，防糊图
        order = np.argsort(-s)[: max_pred]
        b, s, l = b[order], s[order], l[order]
    return b, s, l, max_score, all_scores


def fit_height(panel, target_h):
    h, w = panel.shape[:2]
    return cv2.resize(panel, (max(1, int(round(w * target_h / h))), target_h),
                      interpolation=cv2.INTER_AREA)


def with_caption(panel, text):
    """面板顶部接一条实白说明条（说明不清的根因是叠在图上 + 描边糊字）。"""
    return np.concatenate([band(panel.shape[1], text), panel], axis=0)


def legend_items(score_thresh, max_noise_draw, num_proposals, names, present):
    """图例条目（只列本图真实出现过的类别，避免与面板里的类名对不上）。"""
    items = [("box1", NOISE_COLOR,
              f"row A all proposals ({max_noise_draw}/{num_proposals})"),
             ("box", HILITE_COLOR, "row A head's top-6"),
             ("text", None, "row B x0 pred:")]
    for l in present[:4]:
        items.append(("box", CLASS_COLORS[l % len(CLASS_COLORS)],
                      names[l] if l < len(names) else str(l)))
    if len(present) > 4:
        items.append(("text", None, f"(+{len(present) - 4} more)"))
    items.append(("box", GT_COLOR, f"GT | score>{score_thresh:.2f}"))
    return items


def legend_width(items):
    w = 10
    for kind, _, label in items:
        w += (25 + text_w(label) + 26) if kind != "text" else (text_w(label) + 18)
    return w + 10


def draw_legend(width, items):
    lg = np.full((LEG_H, width, 3), BAND_BG, np.uint8)
    x, y0, y1 = 10, 13, 31
    for kind, color, label in items:
        if kind == "text":
            put_text(lg, label, (x - 8, y1 - 5), scale=0.55)
            x += text_w(label) + 18
        else:
            cv2.rectangle(lg, (x, y0), (x + 18, y1), color, -1 if kind == "box" else 1)
            put_text(lg, label, (x + 25, y1 - 5), scale=0.55)
            x += 25 + text_w(label) + 26
    return lg


# --------------------------------------------------------------------------
def resolve_names(cfg, num_classes):
    """类别名（用于标签文字）。类别数与模型不符时退回占位名。"""
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
    return tc


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

    names = resolve_names(cfg, model.num_classes)
    fname, gt_anns, ds = pick_image(args, cfg)

    # NFE 预算口径与主矩阵（scripts/run_m2_matrix.py）对齐：heun/edm_heun 的
    # sampling_timesteps 是二阶步数 k，实际 NFE≈2k-1，必须折算 k=(B+1)//2；
    # 否则同一个 --steps 下 heun 的 NFE 会是其他求解器的约两倍，横向对比不公平。
    from run_m2_matrix import steps_for_budget
    k_steps, nfe_nominal = steps_for_budget(model.solver_name, args.steps)
    model.sampling_timesteps = k_steps

    print(f"[vis] image = {fname}  (dataset={ds}, gt={len(gt_anns)})")
    print(f"[vis] solver = {model.solver_name}  NFE预算 = {args.steps} -> k={k_steps}"
          f"（名义 NFE≈{nfe_nominal}，实际以计数器为准）  "
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
    results, snapshots, images_whwh = run_with_snapshots(model, batched_inputs, seed=args.seed)

    n_steps = len(snapshots)
    if n_steps == 0:
        raise RuntimeError("没有记录到任何去噪步，采样器未调用 denoise_fn？")

    print(f"\n[vis] 去噪步骤（solver={model.solver_name}, 真实 NFE={n_steps}, "
          f"score>{args.score_thresh}）")
    print(f"{'#':>3} {'t/sigma':>9} {'n_box':>6} {'max_score':>10}")

    # ---------------- 拼图 ----------------
    # 轮廓粗细按"缩放到 panel_h 之后仍有 ~3.5px"折算，避免缩完变成细线
    thick = max(3, int(round(3.5 * image_rgb.shape[0] / args.panel_h)))
    ratio = args.panel_h / image_rgb.shape[0]          # 面板缩放比（写标签用）

    # 结果图落盘用（全分辨率、不缩放）：线宽按图像自身高度定，与原图叠加也清楚
    orig_rgb = img_bgr[:, :, ::-1].copy()
    orig_scale = orig_rgb.shape[1] / image_rgb.shape[1]   # resize 的比例（等比，x/y 相同）
    save_thick = max(3, int(round(0.0045 * image_rgb.shape[0])))
    orig_thick = max(3, int(round(0.0045 * orig_rgb.shape[0])))
    ckpt_tag = "/".join(os.path.normpath(args.ckpt).split(os.sep)[-2:])
    saver = ResultSaver(args.save_steps_dir, header_lines=[
        f"- 配置：`{args.config_file}`",
        f"- 权重：`{args.ckpt}`",
        f"- 求解器：`{model.solver_name}`，真实 NFE={n_steps}，阈值={args.score_thresh}，seed={args.seed}",
        f"- 图像：`{fname}`（原始 {orig_rgb.shape[1]}×{orig_rgb.shape[0]}，"
        f"模型输入 {image_rgb.shape[1]}×{image_rgb.shape[0]}）",
    ])
    if saver.enabled:
        print(f"[vis] 逐步骤结果图 -> {os.path.abspath(args.save_steps_dir)}"
              f"（全分辨率；{'含 x_t；' if args.save_noise else ''}"
              f"{'另存原图尺寸' if args.save_orig_size else '模型输入尺寸'}）")

    def save_pair(boxes, scores, labels, filename, info, with_gt=False):
        # 按求解器分子目录：同一批实验会跑多个 solver，扁平存放会互相覆盖
        save_pred_pair(saver, filename, info, image_rgb, boxes, scores, labels, names,
                       args.label_topk, save_thick, sub=model.solver_name,
                       orig_rgb=orig_rgb, orig_thick=orig_thick,
                       orig_scale=orig_scale, save_orig=args.save_orig_size,
                       gt_xyxy=gt_xyxy if with_gt else None)

    colA, colB, capA, capB, lblB = [], [], [], [], []
    colA.append(draw_gt_panel(image_rgb, gt_xyxy, thick))
    capA.append(f"GT  n={len(gt_xyxy)}")
    saver.add(draw_gt_panel(image_rgb, gt_xyxy, save_thick), "00_gt.jpg",
              f"GT 参考（共用，n={len(gt_xyxy)}）")

    inst_final = results[0]["instances"]
    all_s_final = inst_final.scores.cpu().numpy()
    max_final = float(all_s_final.max()) if len(all_s_final) else 0.0
    fin = inst_final[all_s_final > args.score_thresh]
    fb = fin.pred_boxes.tensor.cpu().numpy()
    fs = fin.scores.cpu().numpy()
    fl = fin.pred_classes.cpu().numpy()
    if len(fs) > args.max_pred:
        order = np.argsort(-fs)[: args.max_pred]
        fb, fs, fl = fb[order], fs[order], fl[order]
    panel = draw_pred_boxes(image_rgb, fb, fs, fl, thick)
    for x1, y1, x2, y2 in clip_xyxy(gt_xyxy, image_rgb.shape[1], image_rgb.shape[0]):
        cv2.rectangle(panel, (int(x1), int(y1)), (int(x2), int(y2)), GT_COLOR, thick - 1)
    colB.append(panel)
    lblB.append((fb, fs, fl))
    capB.append(f"final n={len(fs)} (topk+NMS)")   # 文案须短于说明条宽度，否则被截断
    save_pair(fb, fs, fl, f"final_topkNMS_n{len(fs)}.jpg",
              f"最终检测（topk+NMS，绿框=GT）n={len(fs)} max={max_final:.2f}",
              with_gt=True)

    for k, snap in enumerate(snapshots):
        t_lab = f"t={snap['t']:.0f}" if model.formulation != "edm" else f"sd={snap['t']:.1f}"
        xb = norm_to_abs_xyxy(snap["x_in"], images_whwh, model.scale)[0].cpu().numpy()
        pb, ps, pl, mx, all_s = step_boxes(model, snap["coord"], snap["logits"],
                                           args.score_thresh, args.max_pred)
        print(f"{k + 1:>3} {snap['t']:>9.2f} {len(ps):>6} {mx:>10.3f}")
        colA.append(draw_noise_panel(image_rgb, xb, all_s, args.max_noise_draw, thick))
        capA.append(f"x_t #{k + 1}/{n_steps}  {t_lab}")
        colB.append(draw_pred_boxes(image_rgb, pb, ps, pl, thick))
        lblB.append((pb, ps, pl))
        capB.append(f"x0 #{k + 1}/{n_steps}  n={len(ps)}  max={mx:.2f}")

        # 该步的检测结果图（全分辨率落盘）
        info = (f"solver={model.solver_name} 步骤 {k + 1}/{n_steps} {t_lab} "
                f"n={len(ps)} max={mx:.2f}")
        save_pair(pb, ps, pl, f"step{k + 1:02d}_x0_n{len(ps)}_max{mx:.2f}.jpg", info)
        if args.save_noise:
            saver.add(draw_noise_panel(image_rgb, xb, all_s, args.max_noise_draw,
                                       save_thick),
                      f"step{k + 1:02d}_xt_{t_lab.replace('=', '')}.jpg",
                      info + "（噪声框 x_t）", sub=model.solver_name)

    # 缩放 -> 写标签（在缩放后的面板上写才看得清）-> 顶部接说明条
    present = sorted({int(v) for (_, _, ll) in lblB for v in ll})
    A = [with_caption(fit_height(p, args.panel_h), c) for p, c in zip(colA, capA)]
    B = []
    for p, c, (lb, ls, ll) in zip(colB, capB, lblB):
        p = fit_height(p, args.panel_h)
        p = add_pred_labels(p, lb, ls, ll, names, ratio, args.label_topk)
        B.append(with_caption(p, c))

    cols = len(A)
    nblk = max(1, -(-cols // max(1, args.cols_per_block)))
    step = max(1, -(-cols // nblk))   # 均衡分块，避免最后一块只放一列

    ckpt_tag = "/".join(os.path.normpath(args.ckpt).split(os.sep)[-2:])   # 带上一级目录
    header_text = (
        f"DiffusionDet denoising steps on SDD | {os.path.basename(args.config_file)}"
        f" | solver={model.solver_name} NFE={n_steps}"
        f" | img={os.path.basename(fname)} | seed={args.seed}"
        f" | ckpt={ckpt_tag} | thr={args.score_thresh}")
    band_a_text = ("ROW A - input x_t : noisy proposals fed to the head "
                   "(faint white = all %d; ORANGE-RED = head's top-6)"
                   % min(args.max_noise_draw, model.num_proposals))
    band_b_text = ("ROW B - model output x0 : denoised boxes at this step "
                   "(thick outline; label = class + score)")

    # 画布宽度必须同时容纳「所有面板」与「最长的一行文字」：文字条如果先按
    # 块宽生成，长文字会被 cv2.putText 直接截断（NFE=1 只有 2 列时最容易踩）。
    items = legend_items(args.score_thresh, args.max_noise_draw,
                         model.num_proposals, names, present)
    nat_text = max(text_w(t) + 40 for t in (header_text, band_a_text, band_b_text))
    width = max(min(step, cols) * A[0].shape[1], nat_text, legend_width(items))

    blocks = []
    for s in range(0, cols, step):
        rowA = np.concatenate(A[s:s + step], axis=1)
        rowB = np.concatenate(B[s:s + step], axis=1)
        # 末块列数较少 -> 用统一底色补位（白色会看着像缺图）
        pad = lambda r: np.pad(r, ((0, 0), (0, width - r.shape[1]), (0, 0)),
                               constant_values=BAND_BG)
        rowA, rowB = pad(rowA), pad(rowB)
        band_a = band(width, band_a_text, height=ROW_H, bg=BAND_BG, scale=0.6)
        band_b = band(width, band_b_text, height=ROW_H, bg=BAND_BG, scale=0.6)
        blocks.append(np.concatenate([band_a, rowA, band_b, rowB], axis=0))

    sep = np.full((12, width, 3), 215, np.uint8)
    head = band(width, header_text, height=32, bg=BAND_BG, scale=0.6)
    foot = draw_legend(width, items)

    body = [head]
    for blk in blocks:
        body += [sep, blk]
    body += [sep, foot]
    fig = np.concatenate(body, axis=0)

    os.makedirs(args.out, exist_ok=True)
    name = args.name or f"sdd_diffusion_steps_{model.solver_name}_nfe{n_steps}.jpg"
    out_path = os.path.join(args.out, name)
    cv2.imwrite(out_path, fig[:, :, ::-1])
    print(f"\n[vis] saved -> {out_path}  ({fig.shape[1]}x{fig.shape[0]})")
    saver.close()
    if saver.enabled:
        print(f"[vis] 结果图 {len(saver.rows)} 张 -> {os.path.abspath(args.save_steps_dir)}"
              f"（含 README.md 索引）")


if __name__ == "__main__":
    main()
