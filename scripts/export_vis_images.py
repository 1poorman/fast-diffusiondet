# -*- coding: utf-8 -*-
"""把可视化用到（以及可能要用到）的 SDD 原图统一导出到一个文件夹，便于后期编辑。

为什么需要单独导出：可视化脚本是从数据集目录（datasets/Strawberry/val2017）**现场读图**
的，而 SDD 的 val 集有数百张；做图时反复去数据集里翻很不方便，也不利于事后改图
（想换一张图得重新查文件名）。这里导出两类：

  1. used —— 脚本实际用过的图（与可视化脚本的选图规则完全一致，可复现）；
  2. pool —— 每个类别下按「主类实例数」排序的前若干张候选，方便手工替换。

同时生成 README.md：每张图的原路径、尺寸、主类别、各实例类别/框，以及替换方式。
图片按 `类别名/文件名` 归档，类别名来自 SDD 的 thing_classes，可直接看图挑样本。

用法：
    FASTDD_DATASET=$PWD/datasets ~/.conda/envs/fastdiff/bin/python \\
        scripts/export_vis_images.py --out datasets/vis_images --pool 3
"""
import argparse
import json
import os
import shutil
import sys

import cv2
import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_THIS))
sys.path.insert(0, _THIS)

from train_net import setup  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="导出可视化用的 SDD 原图 + 说明文档")
    p.add_argument("--config-file", default="configs/lab/sdd.res50.bs8.yaml")
    p.add_argument("--out", default="datasets/vis_images", help="导出目录")
    p.add_argument("--ds", default=None, help="数据集名，默认 cfg.DATASETS.TEST[0]")
    p.add_argument("--pool", type=int, default=3, help="每类额外导出前 N 张候选")
    p.add_argument("--no-copy", action="store_true", help="只写文档，不复制图片")
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
    from diffusiondet.data_register import register_all
    register_all(cfg=cfg, verbose=False)

    from detectron2.data import DatasetCatalog, MetadataCatalog
    ds = args.ds or cfg.DATASETS.TEST[0]
    names = list(getattr(MetadataCatalog.get(ds), "thing_classes", []) or [])
    dicts = [d for d in DatasetCatalog.get(ds) if d.get("annotations")]
    root = getattr(cfg, "DATASET_ROOT", "") or os.environ.get("FASTDD_DATASET", "")

    # 每张图：主类别 + 各类实例分布
    recs = []
    for d in dicts:
        cnt = {}
        for a in d["annotations"]:
            cnt[a["category_id"]] = cnt.get(a["category_id"], 0) + 1
        dom = max(cnt, key=cnt.get)
        recs.append({"file": d["file_name"], "dom": dom, "cnt": cnt,
                     "boxes": [{"cat": a["category_id"], "bbox": a["bbox"],
                                "iscrowd": a.get("iscrowd", 0)}
                               for a in d["annotations"]]})

    by_cat = {}
    for r in recs:
        by_cat.setdefault(r["dom"], []).append(r)
    for c in by_cat:
        by_cat[c].sort(key=lambda r: (-r["cnt"][c], -len(r["boxes"])))

    # 与可视化脚本一致的"代表图"：主类平均框面积最大（确定性、不看模型输出）
    used = {}
    for c, rs in by_cat.items():
        def mean_area(r):
            bs = [b["bbox"] for b in r["boxes"] if b["cat"] == c]
            return float(np.mean([b[2] * b[3] for b in bs])) if bs else 0.0
        used[c] = max(rs, key=lambda r: (mean_area(r), r["cnt"][c]))

    os.makedirs(args.out, exist_ok=True)
    manifest = {"dataset": ds, "dataset_root": root, "used": {}, "pool": {}}

    def safe(name):
        return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name)

    for c, r in sorted(used.items()):
        cname = names[c] if c < len(names) else str(c)
        sub = os.path.join(args.out, f"{c:02d}_{safe(cname)}")
        os.makedirs(sub, exist_ok=True)

        picks = [("used", r)] + [(f"pool{i + 1}", x) for i, x in
                                 enumerate(by_cat[c][: args.pool])]
        seen = set()
        entries = []
        for tag, x in picks:
            src = x["file"]
            if src in seen:
                continue
            seen.add(src)
            base = f"{tag}_{os.path.basename(src)}"
            dst = os.path.join(sub, base)
            if not args.no_copy and not os.path.exists(dst):
                shutil.copy2(src, dst)
            im = cv2.imread(src)
            h, w = im.shape[:2] if im is not None else (-1, -1)
            entries.append({
                "tag": tag, "file": base, "src": os.path.abspath(src),
                "size": [w, h],
                "n_instances_total": len(x["boxes"]),
                "instances": {names[k] if k < len(names) else str(k): v
                              for k, v in sorted(x["cnt"].items())},
                "boxes": x["boxes"],
            })
        manifest["used"][cname] = entries[0]
        manifest["pool"][cname] = entries

    # ---------------- README ----------------
    lines = [
        "# SDD 可视化原图",
        "",
        f"- 数据集：`{ds}`（类别顺序即 SDD 的 `thing_classes`）",
        f"- 数据根目录：`{root}`",
        f"- 导出目标：`{os.path.abspath(args.out)}`",
        "",
        "每个类别一个子目录，目录内：",
        "- `used_*.jpg`：可视化脚本自动选出的代表图（规则 = 主类平均框面积最大，",
        "  **确定性、不依赖模型输出**，因此不构成按模型表现挑样本）；",
        "- `poolN_*.jpg`：该类按主类实例数排序的前若干候选，用于手工替换。",
        "",
        "## 各图片清单",
        "",
    ]
    for c, r in sorted(used.items()):
        cname = names[c] if c < len(names) else str(c)
        lines.append(f"### {c}. {cname}")
        lines.append("")
        lines.append("| tag | 文件 | 尺寸 | 实例 | 主类 | 原始路径 |")
        lines.append("|---|---|---|---|---|---|")
        for e in manifest["pool"][cname]:
            inst = ", ".join(f"{k}×{v}" for k, v in e["instances"].items())
            mark = "**主类**" if e["tag"] == "used" else ""
            lines.append(
                f"| {e['tag']} | `{e['file']}` | {e['size'][0]}×{e['size'][1]} | "
                f"{inst} | {mark} | `{e['src']}` |")
        lines.append("")

    lines += [
        "## 后续编辑方式",
        "",
        "1. **换图**：把想用的图片路径传给可视化脚本，可一次给多张：",
        "   ```bash",
        "   FASTDD_DATASET=$PWD/datasets CUDA_VISIBLE_DEVICES=1 \\",
        "     ~/.conda/envs/fastdiff/bin/python scripts/visualize_diffusion_matrix.py \\",
        "       --config-file configs/lab/sdd.res50.bs8.yaml \\",
        "       --ckpt output/lab/sdd.res50.bs8/model_final.pth \\",
        "       --image datasets/vis_images/03_Blossom_Blight/pool1_xxx.jpg --no-traj",
        "   ```",
        "   单图的去噪全过程用 `scripts/visualize_diffusion_steps.py --image <路径>`。",
        "2. **只改版式/阈值**：`--score-thresh`（默认 0.5）、`--panel-h`（面板高度）、",
        "   `--label-topk`（每格最多标注几个框）、`--max-noise-draw`（噪声框数量）。",
        "3. **替换某类的代表图**：改本目录里的 `used_*` 文件后，脚本的自动选图规则不变；",
        "   要让脚本也换图，请显式用 `--image` 指定（脚本本身不读取本目录）。",
        "4. **标注框坐标**：`manifest.json` 保存了每张图的全部实例（`[x, y, w, h]`，"
        "COCO 格式），可直接用于叠画自己的标注或做误差分析。",
        "",
    ]

    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.out, "README.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    n_img = sum(1 for c in by_cat for _ in os.listdir(
        os.path.join(args.out, f"{c:02d}_{safe(names[c] if c < len(names) else str(c))}")))
    print(f"[export] 类别数={len(used)}  图片数={n_img}  目录={os.path.abspath(args.out)}")
    print(f"[export] manifest -> {os.path.join(os.path.abspath(args.out), 'manifest.json')}")
    print(f"[export] 说明文档 -> {os.path.join(os.path.abspath(args.out), 'README.md')}")


if __name__ == "__main__":
    main()
