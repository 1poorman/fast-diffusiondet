# -*- coding: utf-8 -*-
# 多类别图 x 多求解器 x 多 checkpoint 的去噪可视化
# ============================================================================
# 概念澄清：求解器（ddim/euler/heun/dpm_v3）是**推理期**算法，不产生各自的权重；
#   同一个 checkpoint 可被任意 solver 驱动。因此「不同求解器获得的权重」按
#   「不同训练配方得到的 checkpoint」理解：本仓库现有两个（bs8 单卡 = D4 主矩阵
#   基线，bs16 双卡 DDP），两者都用上，并各自跑遍所有 solver。
#
# 产出：
#   1) matrix_final_<B>nfe.jpg —— 行 = (ckpt x solver)，列 = 每个类别一张代表图，
#      每格是该配置的最终检测（绿框=GT），格内标注真实 NFE / 检出数 / 最高分 / GT 数；
#      一张图看全「类别 x 求解器 x 权重」的整体表现与漏检分布。
#   2) traj_<类别>_<B>nfe.jpg —— 每张图一张去噪轨迹：行 = solver，
#      列 = GT + 逐步 x0 预测（相同 NFE 预算，heun 按 k=(B+1)//2 折算，口径同主矩阵）。
#
# 运行：
#   FASTDD_DATASET=$PWD/datasets CUDA_VISIBLE_DEVICES=1 \
#   ~/.conda/envs/fastdiff/bin/python scripts/visualize_diffusion_matrix.py \
#       --config-file configs/lab/sdd.res50.bs8.yaml \
#       --ckpt output/lab/sdd.res50.bs8/model_final.pth \
#              output/lab/sdd.res50.bs16.ddp2/model_final.pth \
#       --solvers ddim euler heun dpm_v3 --steps 10 --per-class \
#       --out results/figures/vis
import argparse
import os
import sys

import cv2
import numpy as np
import torch

_THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_THIS))
sys.path.insert(0, _THIS)

from train_net import Trainer, setup  # noqa: E402
from run_m2_matrix import steps_for_budget  # noqa: E402  NFE 预算口径与主矩阵一致
import visualize_diffusion_steps as V  # noqa: E402  复用推理原语与渲染风格


def parse_args():
    p = argparse.ArgumentParser(description="多类别图 x 多求解器 x 多 checkpoint 的去噪可视化")
    p.add_argument("--config-file", default="configs/lab/sdd.res50.bs8.yaml")
    p.add_argument("--ckpt", nargs="+", required=True, help="一个或多个 checkpoint")
    p.add_argument("--ckpt-tags", nargs="*", default=None, help="每个 ckpt 的短标签")
    p.add_argument("--solvers", nargs="+", default=["ddim", "euler", "heun", "dpm_v3"])
    p.add_argument("--steps", type=int, default=10, help="NFE 预算（heun 自动折算 k）")
    p.add_argument("--per-class", action="store_true", help="每个类别各取一张代表性图")
    p.add_argument("--num-images", type=int, default=7, help="最多取几张图")
    p.add_argument("--image", nargs="*", default=None, help="直接指定图像路径")
    p.add_argument("--score-thresh", type=float, default=0.5)
    p.add_argument("--max-pred", type=int, default=60)
    p.add_argument("--label-topk", type=int, default=4)
    p.add_argument("--panel-h", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-traj", action="store_true", help="跳过逐图去噪轨迹图")
    p.add_argument("--save-steps-dir", default="",
                   help="把每个 NFE 步骤的检测结果图按全分辨率落盘到该目录，"
                        "按 <ckpt>/<solver>/<图像>/ 分层，并生成 README 索引")
    p.add_argument("--save-orig-size", action="store_true",
                   help="结果图另存一份到原始图像分辨率")
    p.add_argument("--out", default="results/figures/vis")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--opts", nargs="*", default=[])
    return p.parse_args()


def pad_w(img, w):
    return np.pad(img, ((0, 0), (0, max(0, w - img.shape[1])), (0, 0)),
                  constant_values=V.BAND_BG)


def hstack(cells):
    return np.concatenate(cells, axis=1)


def pick_per_class(cfg, ds, num_images):
    """按「图像主类别」各挑一张代表图。

    主类别 = 该图中实例数最多的类（SDD 一张图常混多个类）。
    选图准则取【主类别平均框面积最大】：目标物更大更清楚，结论更容易读出，
    且完全不依赖模型输出 —— 不用「按模型表现挑样本」避免 cherry-pick，
    个别类别若仍然漏检，图里如实显示 n=0，需要换图时用 --image 手动指定。
    """
    from detectron2.data import DatasetCatalog, MetadataCatalog

    dicts = [d for d in DatasetCatalog.get(ds) if len(d.get("annotations", [])) > 0]
    names = list(getattr(MetadataCatalog.get(ds), "thing_classes", []) or [])

    buckets = {}
    for d in dicts:
        cnt = {}
        for a in d["annotations"]:
            cnt[a["category_id"]] = cnt.get(a["category_id"], 0) + 1
        dom = max(cnt, key=cnt.get)
        areas = [a["bbox"][2] * a["bbox"][3] for a in d["annotations"]
                 if a["category_id"] == dom]
        buckets.setdefault(dom, []).append(
            (float(np.mean(areas)) if areas else 0.0, cnt[dom], d))

    out = []
    for cid in sorted(buckets):
        cands = sorted(buckets[cid], key=lambda x: (x[0], x[1]), reverse=True)
        area, n_dom, d = cands[0]
        print(f"[matrix] {names[cid] if cid < len(names) else cid}: "
              f"{os.path.basename(d['file_name'])} (主类均框面积={area:.0f}, n={n_dom})")
        out.append((names[cid] if cid < len(names) else str(cid), d))
        if len(out) >= num_images:
            break
    return out


def load_images(cfg, picked):
    """(label, dict) -> 含 resize 后图像/GT 的条目列表（前处理与 detector.forward 一致）。"""
    from detectron2.data import transforms as T

    aug = T.ResizeShortestEdge([cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MAX_SIZE_TEST],
                              cfg.INPUT.MAX_SIZE_TEST)
    items = []
    for label, d in picked:
        img_bgr = cv2.imread(d["file_name"])
        if img_bgr is None:
            print(f"[matrix] 跳过读不到的图：{d['file_name']}")
            continue
        h0, w0 = img_bgr.shape[:2]
        tfm = aug.get_transform(img_bgr)
        image = tfm.apply_image(img_bgr)
        gt = np.zeros((0, 4), np.float32)
        if d.get("annotations"):
            raw = np.array([[a["bbox"][0], a["bbox"][1],
                             a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]]
                            for a in d["annotations"]], dtype=np.float32)
            gt = tfm.apply_box(raw).astype(np.float32)
        items.append({
            "label": label,
            "file": d["file_name"],
            "gt": gt,
            "rgb": image[:, :, ::-1].copy(),
            "rgb_orig": img_bgr[:, :, ::-1].copy(),
            "tensor": torch.as_tensor(image.astype("float32").transpose(2, 0, 1)),
            "hw": (h0, w0),
        })
    return items


def matrix_legend_items(score_thresh, names, present):
    """本图专用图例条目（矩阵图没有噪声行，不能套用带 ROW A 文案的通用图例）。"""
    items = [("text", None, "detection:")]
    for c in present:
        items.append(("box", V.CLASS_COLORS[c % len(V.CLASS_COLORS)],
                      names[c] if c < len(names) else str(c)))
    items.append(("box", V.GT_COLOR, f"GT | score>{score_thresh:.2f}"))
    items.append(("text", None, "| label = class + score"))
    return items


def caption_band(text, width, bg=(255, 255, 255)):
    """按可用宽度自动缩小字号的说明条，避免长文案被 cv2.putText 截断。"""
    for scale in (0.58, 0.54, 0.50, 0.46, 0.42, 0.38):
        if V.text_w(text, scale) + 16 <= width:
            break
    return V.band(width, text, height=V.CAP_H, bg=bg, scale=scale)


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
    cfg.MODEL.DEVICE = f"cuda:{args.gpu}"
    cfg.MODEL.DiffusionDet.BOX_RENEWAL = False      # 与 FEP 主矩阵口径一致
    cfg.MODEL.DiffusionDet.USE_ENSEMBLE = False
    cfg.MODEL.DiffusionDet.SAMPLE_STEP = args.steps
    if cfg.MODEL.DiffusionDet.FORMULATION == "edm":
        cfg.MODEL.DiffusionDet.SOLVER = "edm_heun"
    if args.opts:
        cfg.merge_from_list(args.opts)
    cfg.freeze()

    from diffusiondet.data_register import register_all
    register_all(cfg=cfg, verbose=False)
    ds = cfg.DATASETS.TEST[0]

    # ---------------- 选图 ----------------
    if args.image:
        from detectron2.data import DatasetCatalog
        by_path = {os.path.abspath(d["file_name"]): d for d in DatasetCatalog.get(ds)}
        picked = []
        for p in args.image:
            ap = os.path.abspath(p)
            picked.append((os.path.basename(ap).rsplit(".", 1)[0], by_path.get(ap)
                           or {"file_name": ap, "annotations": []}))
    elif args.per_class:
        picked = pick_per_class(cfg, ds, args.num_images)
    else:
        from detectron2.data import DatasetCatalog
        cand = [d for d in DatasetCatalog.get(ds) if len(d.get("annotations", [])) > 0]
        picked = [("img0", cand[0])]
    items = load_images(cfg, picked)
    print("[matrix] 选用图像：" + ", ".join(
        f"{it['label']}({os.path.basename(it['file'])})" for it in items))

    tags = args.ckpt_tags or ["/".join(os.path.normpath(c).split(os.sep)[-2:])
                              for c in args.ckpt]
    assert len(tags) == len(args.ckpt), "--ckpt-tags 数量必须与 --ckpt 一致"

    # ---------------- 逐 ckpt、逐 solver 推理 ----------------
    from detectron2.checkpoint import DetectionCheckpointer

    panels = {}      # (ci, solver, i) -> (面板, 真实NFE, 检出数, 最高分)
    traj = {}        # (solver, i) -> [(x0 面板, 说明)]，只对第一个 ckpt 出轨迹图
    present_ids = set()   # 图中真实出现过的类别（图例只列这些）
    class_names = None
    saver = V.ResultSaver(args.save_steps_dir, header_lines=[
        f"- 配置：`{args.config_file}`",
        f"- 权重：{', '.join('`%s`' % c for c in args.ckpt)}",
        f"- 求解器：{', '.join(args.solvers)}；NFE 预算={args.steps}"
        f"（heun 按 k=(B+1)//2 折算，真实 NFE 见行内标注）",
        f"- 阈值={args.score_thresh}，seed={args.seed}",
        "- 目录结构：`<ckpt标签>/<solver>/<图像名>/`，每个文件是**未经缩放**的检测结果图",
    ])
    if saver.enabled:
        print(f"[matrix] 逐步骤结果图 -> {os.path.abspath(args.save_steps_dir)}")
    for ci, (ckpt, tag) in enumerate(zip(args.ckpt, tags)):
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir="").resume_or_load(ckpt, resume=False)
        model.eval()
        class_names = V.resolve_names(cfg, model.num_classes)
        for solver in args.solvers:
            k, _ = steps_for_budget(solver, args.steps)
            model.solver_name = solver                 # forward 期读取，直接改写即可
            model.sampling_timesteps = k
            for i, item in enumerate(items):
                batched = [{"image": item["tensor"], "height": item["hw"][0],
                            "width": item["hw"][1]}]
                results, snaps, _ = V.run_with_snapshots(model, batched, seed=args.seed)
                thick = max(3, int(round(3.5 * item["rgb"].shape[0] / args.panel_h)))
                ratio = args.panel_h / item["rgb"].shape[0]

                inst = results[0]["instances"]
                all_scores = inst.scores.cpu().numpy()
                # 最高分取【全部 proposal】：过阈后为空时，仍能看出"离阈值差多远"
                max_all = float(all_scores.max()) if len(all_scores) else 0.0
                fin = inst[all_scores > args.score_thresh]
                b = fin.pred_boxes.tensor.cpu().numpy()
                s = fin.scores.cpu().numpy()
                l = fin.pred_classes.cpu().numpy()
                if len(s) > args.max_pred:
                    o = np.argsort(-s)[: args.max_pred]
                    b, s, l = b[o], s[o], l[o]

                panel = V.draw_pred_boxes(item["rgb"], b, s, l, thick)
                for x1, y1, x2, y2 in V.clip_xyxy(item["gt"], item["rgb"].shape[1],
                                                  item["rgb"].shape[0]):
                    cv2.rectangle(panel, (int(x1), int(y1)), (int(x2), int(y2)),
                                  V.GT_COLOR, max(2, thick - 1))
                panel = V.add_pred_labels(V.fit_height(panel, args.panel_h),
                                          b, s, l, class_names, ratio, args.label_topk)
                panels[(ci, solver, i)] = (panel, len(snaps), len(s), max_all)
                present_ids.update(int(v) for v in l)

                # ---- 逐 NFE 步骤的检测结果图（全分辨率落盘，便于后期编辑）----
                if saver.enabled:
                    stem = os.path.splitext(os.path.basename(item["file"]))[0]
                    sub = f"{tag}/{solver}/{stem}"
                    st = max(3, int(round(0.0045 * item["rgb"].shape[0])))
                    ot = max(3, int(round(0.0045 * item["rgb_orig"].shape[0])))
                    os_ = item["rgb_orig"].shape[1] / item["rgb"].shape[1]
                    base = dict(sub=sub, orig_rgb=item["rgb_orig"], orig_thick=ot,
                                orig_scale=os_, save_orig=args.save_orig_size)
                    V.save_pred_pair(saver, f"final_topkNMS_n{len(s)}.jpg",
                                     f"solver={solver} NFE={len(snaps)} 最终检测 "
                                     f"（topk+NMS，绿框=GT）n={len(s)} max={max_all:.2f}",
                                     item["rgb"], b, s, l, class_names,
                                     args.label_topk, st, gt_xyxy=item["gt"], **base)
                    for kk, snap in enumerate(snaps):
                        pb, ps, pl, mx_k, _ = V.step_boxes(model, snap["coord"],
                                                           snap["logits"],
                                                           args.score_thresh, args.max_pred)
                        V.save_pred_pair(saver,
                                         f"step{kk + 1:02d}_x0_n{len(ps)}_max{mx_k:.2f}.jpg",
                                         f"solver={solver} 步骤 {kk + 1}/{len(snaps)} "
                                         f"n={len(ps)} max={mx_k:.2f}",
                                         item["rgb"], pb, ps, pl, class_names,
                                         args.label_topk, st, **base)

                if ci == 0 and not args.no_traj:
                    caps = []
                    for kk, snap in enumerate(snaps):
                        pb, ps, pl, _, _ = V.step_boxes(model, snap["coord"], snap["logits"],
                                                        args.score_thresh, args.max_pred)
                        present_ids.update(int(v) for v in pl)
                        p = V.draw_pred_boxes(item["rgb"], pb, ps, pl, thick)
                        p = V.add_pred_labels(V.fit_height(p, args.panel_h), pb, ps, pl,
                                              class_names, ratio, 2)
                        caps.append((p, f"#{kk + 1}/{len(snaps)} n={len(ps)}"))
                    traj[(solver, i)] = caps
                print(f"[matrix] ckpt={tag} solver={solver} nfe={len(snaps)} "
                      f"img={os.path.basename(item['file'])} n={len(s)} "
                      f"max={panels[(ci, solver, i)][3]:.2f}")
        del model
        torch.cuda.empty_cache()

    present = sorted(present_ids) or [0]

    # ---------------- 图 1：最终检测矩阵（行 = ckpt x solver，列 = 类别图） ----------------
    solvers = args.solvers
    col_w = max(panels[(0, solvers[0], i)][0].shape[1] for i in range(len(items)))
    grid_w = col_w * len(items)
    leg_items = matrix_legend_items(args.score_thresh, class_names, present)

    header_text = ("SDD per-class representatives x solvers x checkpoints"
                   "  (final detection; green = GT)")
    # 画布宽度必须容纳「面板网格」「最长文字行」「图例」三者，否则长文字被截断
    nat = max(V.text_w(header_text, 0.62) + 40, V.legend_width(leg_items))
    width = max(grid_w, nat)

    header = hstack([V.band(col_w, f"{it['label']}  GT={len(it['gt'])}",
                            height=V.CAP_H + 6, bg=V.BAND_BG, scale=0.58)
                     for it in items])
    header = pad_w(header, width)
    blocks = [V.band(width, header_text, height=32, bg=V.BAND_BG, scale=0.62), header]
    for ci, tag in enumerate(tags):
        for solver in solvers:
            # 行内已有 ckpt/solver，格内只留该格的关键数字（长文案会被裁）
            cells = [V.with_caption(
                pad_w(panels[(ci, solver, i)][0], col_w),
                f"n={panels[(ci, solver, i)][2]}  max={panels[(ci, solver, i)][3]:.2f}"
                f"  /GT={len(items[i]['gt'])}")
                for i in range(len(items))]
            row = pad_w(hstack(cells), width)
            band = V.band(width, f"ckpt = {tag}    solver = {solver}"
                                 f"    NFE={panels[(ci, solver, 0)][1]}"
                                 f"    thr={args.score_thresh}",
                          height=V.ROW_H, bg=V.BAND_BG, scale=0.62)
            blocks.append(np.concatenate([band, row], axis=0))

    sep = np.full((12, width, 3), 215, np.uint8)
    body = [blocks[0]]
    for b in blocks[1:]:
        body += [sep, b]
    fig = np.concatenate(body + [sep, V.draw_legend(width, leg_items)], axis=0)

    os.makedirs(args.out, exist_ok=True)
    out1 = os.path.join(args.out, f"matrix_final_{args.steps}nfe.jpg")
    cv2.imwrite(out1, fig[:, :, ::-1])
    print(f"[matrix] saved -> {out1}  ({fig.shape[1]}x{fig.shape[0]})")

    # ---------------- 图 2：每张图一张「行=solver」的去噪轨迹 ----------------
    if not args.no_traj:
        for i, item in enumerate(items):
            thick = max(3, int(round(3.5 * item["rgb"].shape[0] / args.panel_h)))
            gt_cell = V.with_caption(V.fit_height(
                V.draw_gt_panel(item["rgb"], item["gt"], thick), args.panel_h),
                f"GT n={len(item['gt'])}")
            rows = []
            for solver in solvers:
                caps = traj[(solver, i)]
                row = hstack([gt_cell] + [V.with_caption(p, c) for p, c in caps])
                rows.append(np.concatenate(
                    [V.band(row.shape[1],
                            f"solver = {solver}   (NFE={panels[(0, solver, i)][1]})",
                            height=V.ROW_H, bg=V.BAND_BG, scale=0.62), row], axis=0))
            hdr_text = (f"SDD denoising trajectory | {item['label']} | "
                        f"{os.path.basename(item['file'])} | ckpt={tags[0]} | "
                        f"NFE budget={args.steps} | seed={args.seed} | "
                        f"thr={args.score_thresh}")
            width = max(max(r.shape[1] for r in rows),
                        V.text_w(hdr_text, 0.62) + 40,
                        V.legend_width(leg_items))
            rows = [pad_w(r, width) for r in rows]
            sep = np.full((12, width, 3), 215, np.uint8)
            header = V.band(width, hdr_text, height=32, bg=V.BAND_BG, scale=0.62)
            body = [header]
            for r in rows:
                body += [sep, r]
            fig = np.concatenate(body + [sep, V.draw_legend(width, leg_items)], axis=0)
            out2 = os.path.join(args.out, f"traj_{item['label']}_{args.steps}nfe.jpg")
            cv2.imwrite(out2, fig[:, :, ::-1])
            print(f"[matrix] saved -> {out2}  ({fig.shape[1]}x{fig.shape[0]})")

    saver.close()
    if saver.enabled:
        print(f"[matrix] 逐步骤结果图 {len(saver.rows)} 张 -> "
              f"{os.path.abspath(args.save_steps_dir)}（含 README.md 索引）")


if __name__ == "__main__":
    main()
