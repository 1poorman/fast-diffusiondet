"""Fixed-threshold descriptive audit of Q1 predictions, separate from AP-based gates."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.v3_error_audit import audit_image, spearman
from scripts.v3_head_sweep import sha256
from scripts.v3_quality import ARMS
from scripts.v3_summarize_distill import write_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    root = args.run_dir
    manifest = json.loads((root / "manifest.json").read_text())
    if not (root / "COMPLETE.json").is_file() or len(manifest["image_ids"]) != manifest["full_val_images"]:
        raise ValueError("Need a completed full-validation run")
    annotation_path = Path(manifest["args"]["dataset_root"]) / "Strawberry/annotations/instances_val2017.json"
    if sha256(annotation_path) != manifest["annotation_sha256"]["val"]:
        raise ValueError("Annotation hash mismatch")
    data = json.loads(annotation_path.read_text())
    if any(a.get("iscrowd", 0) or a.get("ignore", 0) for a in data["annotations"]):
        raise ValueError("This descriptive audit does not support crowd or ignored objects")
    gt = defaultdict(list)
    for ann in data["annotations"]:
        gt[ann["image_id"]].append(ann)
    rows, bins, hashes = [], [], {}
    for seed in manifest["args"]["train_seeds"]:
        for arm in ARMS:
            for e in manifest["args"]["eval_seeds"]:
                path = root / f"seed{seed}_{arm}/eval_seed{e}/coco_instances_results.json"
                hashes[str(path)] = sha256(path)
                grouped = defaultdict(list)
                for pred in json.loads(path.read_text()):
                    grouped[pred["image_id"]].append(pred)
                for threshold in (0.05, 0.3, 0.5):
                    counts, scores, quality = Counter(), [], []
                    for image_id in manifest["image_ids"]:
                        local, s, q = audit_image(grouped[image_id], gt[image_id], threshold)
                        counts.update(local)
                        scores.extend(s)
                        quality.extend(q)
                    if counts["predictions"] != sum(counts[k] for k in ("matched", "duplicate", "classification", "localization", "background")):
                        raise ValueError("Diagnostic categories do not partition predictions")
                    s, q = np.array(scores), np.array(quality)
                    positive = q >= 0.1
                    rows.append(dict(train_seed=seed, arm=arm, eval_seed=e, threshold=threshold, **counts,
                                     coverage50=counts["covered_gt50"] / counts["gt"], coverage75=counts["covered_gt75"] / counts["gt"],
                                     matched_precision=counts["matched"] / max(1, counts["predictions"]),
                                     score_iou_spearman_on_overlaps=spearman(s[positive], q[positive])))
                    if threshold == 0.05:
                        for lower in np.arange(0, 1, 0.1):
                            mask = (s >= lower) & (s < lower + 0.1)
                            if mask.any():
                                bins.append(dict(train_seed=seed, arm=arm, eval_seed=e, score_bin_lower=float(lower),
                                                 count=int(mask.sum()), mean_score=float(s[mask].mean()),
                                                 mean_same_class_iou=float(q[mask].mean()), fraction_iou75=float((q[mask] >= 0.75).mean())))
    raw = Path("results/raw")
    write_rows(raw / "v3_quality_diagnostics.csv", rows)
    write_rows(raw / "v3_quality_calibration.csv", bins)
    report = dict(annotation_sha256=sha256(annotation_path), prediction_sha256=hashes,
                  script_sha256=sha256(__file__), audit_module_sha256=sha256("scripts/v3_error_audit.py"),
                  thresholds=[0.05, 0.3, 0.5],
                  scope="Descriptive fixed-threshold counts, not TIDE/AP attribution; GT coverage is many-to-one, not COCO AR; no deployment threshold selection")
    (raw / "v3_quality_diagnostics_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    for arm in ARMS:
        selected = [r for r in rows if r["arm"] == arm and r["threshold"] == 0.3]
        print(json.dumps(dict(arm=arm, **{k: float(np.mean([r[k] for r in selected])) for k in
                                        ("predictions", "matched", "background", "localization", "classification", "duplicate", "coverage50", "coverage75")})))


if __name__ == "__main__":
    main()
