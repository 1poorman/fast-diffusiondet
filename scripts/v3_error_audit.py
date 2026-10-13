"""Descriptive post-NMS error audit of frozen G1 predictions, not COCO AP or TIDE.

No oracle labels enter a model. The fixed score thresholds are diagnostics, not
deployment threshold selection. GT coverage is many-to-one, unlike matched AR.
"""
import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.v3_head_sweep import sha256
from scripts.v3_summarize_distill import write_rows


def spearman(x, y):
    def ranks(values):
        _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
        return (np.cumsum(counts) - counts + (counts-1)/2)[inverse]
    a, b = ranks(x), ranks(y)
    return float(np.corrcoef(a, b)[0, 1]) if len(a)>1 and a.std()>0 and b.std()>0 else float("nan")


def box_iou(boxes, targets):
    """COCO xywh boxes, continuous coordinates, no pixel +1 convention."""
    a = np.asarray(boxes, dtype=float).reshape(-1, 4)
    b = np.asarray(targets, dtype=float).reshape(-1, 4)
    area_a = np.maximum(a[:, 2:], 0).prod(-1)
    area_b = np.maximum(b[:, 2:], 0).prod(-1)
    extent = np.minimum(a[:, None, :2]+a[:, None, 2:], b[None, :, :2]+b[None, :, 2:])
    extent -= np.maximum(a[:, None, :2], b[None, :, :2])
    intersection = np.maximum(extent, 0).prod(-1)
    return intersection / np.maximum(area_a[:, None]+area_b[None, :]-intersection, 1e-12)


def audit_image(predictions, targets, threshold):
    preds = sorted((p for p in predictions if p["score"] >= threshold), key=lambda p: -p["score"])
    iou = box_iou([p["bbox"] for p in preds], [t["bbox"] for t in targets])
    same_class = np.array([p["category_id"] for p in preds])[:, None] == np.array([t["category_id"] for t in targets])[None, :]
    same_iou = iou*same_class
    counts = Counter(predictions=len(preds), gt=len(targets), matched=0, duplicate=0, classification=0,
                     localization=0, background=0, covered_gt50=0, covered_gt75=0)
    used = set()
    scores, qualities = [], []
    for j, pred in enumerate(preds):
        overlaps = same_iou[j]
        eligible = [k for k in np.argsort(-overlaps) if overlaps[k] >= 0.5 and k not in used]
        quality = float(overlaps.max(initial=0))
        scores.append(pred["score"])
        qualities.append(quality)
        if eligible:
            used.add(eligible[0])
            counts["matched"] += 1
        elif quality >= 0.5:
            counts["duplicate"] += 1
        elif iou[j].max(initial=0) >= 0.5:
            counts["classification"] += 1
        elif quality >= 0.1:
            counts["localization"] += 1
        else:
            counts["background"] += 1
    if len(preds):
        coverage = same_iou.max(axis=0)
        counts["covered_gt50"] = int((coverage >= 0.5).sum())
        counts["covered_gt75"] = int((coverage >= 0.75).sum())
    return counts, scores, qualities


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, default=Path("runs/v3/head_sweep_20261011"))
    p.add_argument("--annotation", type=Path, default=Path("datasets/Strawberry/annotations/instances_val2017.json"))
    p.add_argument("--out-dir", type=Path, default=Path("runs/v3/error_audit_20261011"))
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    annotation = json.loads(args.annotation.read_text())
    if any(a.get("iscrowd",0) or a.get("ignore",0) for a in annotation["annotations"]):
        raise ValueError("This diagnostic only supports annotations without crowd/ignored objects")
    gt = defaultdict(list)
    for a in annotation["annotations"]:
        gt[a["image_id"]].append(a)
    image_ids = [i["id"] for i in annotation["images"]]
    rows, calibration, source_hashes = [], [], {}
    for depth in (4,6):
        for seed in (0,1,2):
            path = args.run_dir / f"seed{seed}_d{depth}_no_weights/coco_instances_results.json"
            source_hashes[str(path)] = sha256(path)
            grouped = defaultdict(list)
            for pred in json.loads(path.read_text()):
                grouped[pred["image_id"]].append(pred)
            for threshold in (0.05, 0.3, 0.5):
                counts, all_scores, all_quality = Counter(), [], []
                for image_id in image_ids:
                    local, scores, quality = audit_image(grouped[image_id], gt[image_id], threshold)
                    counts.update(local)
                    all_scores.extend(scores)
                    all_quality.extend(quality)
                if counts["predictions"] != sum(counts[k] for k in ("matched","duplicate","classification","localization","background")):
                    raise RuntimeError("Error categories do not partition predictions")
                score, quality = np.asarray(all_scores), np.asarray(all_quality)
                positive = quality >= 0.1
                correlation = spearman(score[positive], quality[positive])
                rows.append(dict(depth=depth, eval_seed=seed, score_threshold=threshold, **counts,
                                 matched_precision=counts["matched"]/max(1,counts["predictions"]),
                                 coverage50=counts["covered_gt50"]/counts["gt"],
                                 coverage75=counts["covered_gt75"]/counts["gt"],
                                 score_iou_spearman_on_overlaps=correlation,
                                 mean_same_class_iou=float(quality.mean())))
                if threshold == 0.05:
                    for lower in np.arange(0, 1, 0.1):
                        mask = (score >= lower) & (score < lower+0.1)
                        if mask.any():
                            calibration.append(dict(depth=depth,eval_seed=seed,score_bin_lower=float(lower),
                                                    count=int(mask.sum()),mean_score=float(score[mask].mean()),
                                                    mean_same_class_iou=float(quality[mask].mean()),
                                                    fraction_iou75=float((quality[mask]>=0.75).mean())))
    write_rows(args.out_dir / "diagnostics.csv", rows)
    write_rows(args.out_dir / "calibration.csv", calibration)
    manifest = dict(annotation_sha256=sha256(args.annotation), prediction_sha256=source_hashes,
                    script_sha256=sha256(__file__), images=len(image_ids), objects=sum(map(len,gt.values())),
                    definition="Post-NMS score-sorted greedy same-class matching at IoU0.5. Remaining errors: duplicate, wrong class IoU>=0.5, same-class 0.1<=IoU<0.5, background. Coverage is many-to-one and not official AR. Correlation uses same-class IoU>=0.1 predictions.",
                    limitation="Descriptive on the reused development set; no AP attribution, no threshold tuning, no causal model-gain estimate.")
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    for name in ("diagnostics.csv", "calibration.csv", "manifest.json"):
        (Path("results/raw") / f"v3_error_audit_{name}").write_bytes((args.out_dir/name).read_bytes())
    for depth in (4,6):
        subset = [r for r in rows if r["depth"]==depth and r["score_threshold"]==0.3]
        print(json.dumps({"depth":depth, **{k:float(np.mean([r[k] for r in subset])) for k in subset[0]
                                          if k not in ("depth","eval_seed","score_threshold")}}, indent=2))
    os.environ.setdefault("MPLCONFIGDIR",str(args.out_dir.resolve()/"matplotlib_cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4.2))
    kinds=["duplicate","classification","localization","background"]
    for j,depth in enumerate((4,6)):
        rs=[r for r in rows if r["depth"]==depth and r["score_threshold"]==0.3]
        axes[0].bar(np.arange(4)+(j-0.5)*0.36,[np.mean([r[k] for r in rs]) for k in kinds],width=0.36,label=f"H{depth}")
        points=[]
        for lower in sorted({r["score_bin_lower"] for r in calibration}):
            bins=[r for r in calibration if r["depth"]==depth and r["score_bin_lower"]==lower]
            if bins:
                n=sum(r["count"] for r in bins)
                points.append((sum(r["mean_score"]*r["count"] for r in bins)/n,
                               sum(r["mean_same_class_iou"]*r["count"] for r in bins)/n))
        axes[1].plot(*zip(*points),marker="o",label=f"H{depth}")
    axes[0].set(xticks=range(4),xticklabels=kinds,ylabel="Mean count / 750 images",title="Post-NMS errors at score >= 0.3")
    axes[0].tick_params(axis="x",labelrotation=15)
    axes[1].set(xlabel="Mean class confidence",ylabel="Mean best same-class GT IoU",title="Score vs box quality (descriptive)")
    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig("results/figures/v3_error_audit.png",dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
