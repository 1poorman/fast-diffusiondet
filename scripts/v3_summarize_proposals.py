"""Audit and archive P1; average inference seeds within each frozen checkpoint."""
import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import shutil
import statistics as stats
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.v3_head_sweep import sha256
from scripts.v3_summarize_distill import read_rows, write_rows


def row_key(row):
    return tuple(int(row[k]) for k in ("train_seed", "eval_seed", "proposals"))


def average(rows, key):
    return stats.mean(float(row[key]) for row in rows)


def audit_matrix(rows, coverage, manifest, completed, parity, reference):
    """Reject partial, duplicate, unpaired, or numerically invalid evidence."""
    args = manifest["args"]
    seeds, eval_seeds, proposals = (args[k] for k in ("train_seeds", "eval_seeds", "proposals"))
    if (set(seeds) != {10, 11, 12} or set(eval_seeds) != {0, 1, 2}
            or set(proposals) != {300, 200, 150, 100}
            or any(len(v) != len(set(v)) for v in (seeds, eval_seeds, proposals))):
        raise ValueError("P1 report requires all predeclared checkpoints, inference seeds, and budgets")
    expected = {(s, e, p) for s in seeds for e in eval_seeds for p in proposals}
    if ({row_key(r) for r in rows} != expected or len(rows) != len(expected)
            or completed != dict(evaluations=len(expected), checkpoints=len(seeds))):
        raise ValueError("Incomplete or duplicate evaluation matrix")
    ids = manifest["image_ids"]
    if args["limit"] != 0 or len(ids) != manifest["full_images"] or len(set(ids)) != len(ids):
        raise ValueError("Need the full validation image set")
    if any(str(r["full_dataset"]) != "True" or int(r["images"]) != len(ids)
           or int(r["measured_images"]) != 200 for r in rows):
        raise ValueError("Need full validation and standard timing")
    metrics = [k for k in rows[0] if k.startswith("AP") or k.startswith("latency_")]
    if any(not math.isfinite(float(r[k])) for r in rows for k in metrics):
        raise ValueError("Non-finite evaluation metrics")
    if any(float(r["latency_p50_ms"]) <= 0
           or float(r["latency_p95_ms"]) < float(r["latency_p50_ms"]) for r in rows):
        raise ValueError("Invalid latency quantiles")
    if {k[3:] for k in metrics if k.startswith("AP-")} != set(manifest["category_names"]):
        raise ValueError("Incomplete category metrics")
    if (len(parity) != 2 * len(seeds)
            or {(r["train_seed"], r["image_id"]) for r in parity} != {(s, i) for s in seeds for i in ids[:2]}
            or not all(r["bit_exact"] is True for r in parity)):
        raise ValueError("Missing P300 bit-exact inference parity")
    reference = [r for r in reference if r["arm"] == "uniform" and int(r["train_seed"]) in seeds]
    ref = {(int(r["train_seed"]), int(r["eval_seed"])): r for r in reference}
    if len(ref) != len(seeds) * len(eval_seeds) or len(ref) != len(reference):
        raise ValueError("Incomplete or duplicate T1 baseline reference")
    baseline_deltas = []
    for row in rows:
        if int(row["proposals"]) != 300:
            continue
        original = ref[(int(row["train_seed"]), int(row["eval_seed"]))]
        deltas = {k: float(row[k]) - float(original[k]) for k in metrics if k.startswith("AP")}
        if any(not math.isfinite(v) or abs(v) > 1e-5 for v in deltas.values()):
            raise ValueError("P300 AP differs from the matching T1 checkpoint evaluation")
        baseline_deltas.append(max(abs(v) for v in deltas.values()))
    groups = {"all", "crowded_ge5", *[f"category_{c}" for c in range(len(manifest["category_names"]))]}
    expected_coverage = {(*key, stage, group) for key in expected for stage in range(1, 5) for group in groups}
    indexed = {(*row_key(r), int(r["stage"]), r["group"]): r for r in coverage}
    if set(indexed) != expected_coverage or len(coverage) != len(expected_coverage):
        raise ValueError("Incomplete or duplicate stage coverage matrix")
    totals = {}
    for row in coverage:
        total, h50, h75 = (int(row[k]) for k in ("objects", "hits50", "hits75"))
        if not 0 <= h75 <= h50 <= total or total == 0:
            raise ValueError("Invalid coverage counts")
        if any(not math.isfinite(float(row[k])) or abs(float(row[k]) - h / total) > 1e-12
               for k, h in (("coverage50", h50), ("coverage75", h75))):
            raise ValueError("Coverage fraction differs from recorded counts")
        if totals.setdefault(row["group"], total) != total:
            raise ValueError("GT denominators differ across stages or budgets")
    if sum(v for k, v in totals.items() if k.startswith("category_")) != totals["all"]:
        raise ValueError("Category counts do not sum to all GT")
    if totals["crowded_ge5"] > totals["all"]:
        raise ValueError("Crowded GT count exceeds total")
    for key in expected:
        for stage in range(1, 5):
            all_gt = indexed[(*key, stage, "all")]
            crowded = indexed[(*key, stage, "crowded_ge5")]
            categories = [indexed[(*key, stage, g)] for g in groups if g.startswith("category_")]
            for field in ("hits50", "hits75"):
                if (sum(int(r[field]) for r in categories) != int(all_gt[field])
                        or int(crowded[field]) > int(all_gt[field])):
                    raise ValueError("Category or crowded coverage hits disagree with all GT")
    return dict(evaluations=len(rows), coverage_rows=len(coverage), gt_counts=totals,
                baseline_max_abs_AP_difference=max(baseline_deltas)), metrics


def summarize(rows, coverage, manifest, metrics):
    paired = []
    for seed in manifest["args"]["train_seeds"]:
        base = [r for r in rows if int(r["train_seed"]) == seed and int(r["proposals"]) == 300]
        for count in manifest["args"]["proposals"]:
            arm = [r for r in rows if int(r["train_seed"]) == seed and int(r["proposals"]) == count]
            row = dict(train_seed=seed, proposals=count)
            for key in metrics:
                row[key] = average(arm, key)
                row[f"baseline_{key}"] = average(base, key)
                row[f"delta_{key}"] = row[key] - row[f"baseline_{key}"]
            row["p50_reduction_percent"] = 100 * (1 - row["latency_p50_ms"] / row["baseline_latency_p50_ms"])
            for group in ("all", "crowded_ge5"):
                for threshold in (50, 75):
                    key = f"coverage{threshold}"
                    selected = [r for r in coverage if int(r["train_seed"]) == seed
                                and int(r["stage"]) == 4 and r["group"] == group]
                    value = average([r for r in selected if int(r["proposals"]) == count], key)
                    baseline = average([r for r in selected if int(r["proposals"]) == 300], key)
                    row[f"{group}_{key}_percent"] = 100 * value
                    row[f"delta_{group}_{key}_pp"] = 100 * (value - baseline)
            paired.append(row)
    summary, per_class, gates = [], [], []
    for count in manifest["args"]["proposals"]:
        selected = [r for r in paired if r["proposals"] == count]
        row = {k: average(selected, k) for k in paired[0] if k != "train_seed"}
        row["proposals"] = count
        row["delta_AP_checkpoint_sample_sd"] = stats.stdev(r["delta_AP"] for r in selected)
        row["AP_checkpoint_sample_sd"] = stats.stdev(r["AP"] for r in selected)
        for category in manifest["category_names"]:
            per_class.append(dict(proposals=count, category=category, AP=row[f"AP-{category}"],
                                  baseline_AP=row[f"baseline_AP-{category}"], delta_AP=row[f"delta_AP-{category}"]))
        worst = min(manifest["category_names"], key=lambda c: row[f"delta_AP-{c}"])
        row["worst_category"] = worst
        row["worst_category_delta_AP"] = row[f"delta_AP-{worst}"]
        summary.append(row)
        if count != 300:
            checks = dict(AP=row["delta_AP"] >= -0.3, AP75=row["delta_AP75"] >= -0.3,
                          p50=row["p50_reduction_percent"] >= 10,
                          worst_category=row["worst_category_delta_AP"] >= -2,
                          crowded_coverage50=row["delta_crowded_ge5_coverage50_pp"] >= -1)
            gates.append(dict(proposals=count, checks=checks, passed_development_screen=all(checks.values())))
    return paired, summary, per_class, gates


def audit_annotations(original, evaluation):
    """T1 hashes raw COCO; P1 hashes the evaluator's background-filtered COCO."""
    categories = [r for r in original["categories"] if r["id"] != 0 and r["name"] != "_background_"]
    kept = {r["id"] for r in categories}
    expected = dict(info=original.get("info", {}), licenses=original.get("licenses", []),
                    images=original["images"], categories=categories,
                    annotations=[r for r in original["annotations"] if r["category_id"] in kept])
    if evaluation != expected:
        raise ValueError("Evaluator annotations differ from background-filtered T1 annotations")


def audit_files(root, manifest):
    import yaml
    for source, digest in manifest["source_sha256"].items():
        if sha256(root / "source_snapshot" / source) != digest:
            raise ValueError(f"Source snapshot hash mismatch: {source}")
    checkpoint_root = Path(manifest["args"]["checkpoint_root"])
    t1 = json.loads((checkpoint_root / "manifest.json").read_text())
    if not (checkpoint_root / "COMPLETE.json").is_file() or t1["eval_image_ids"] != manifest["image_ids"]:
        raise ValueError("T1 reference is incomplete or uses different images")
    if t1["args"]["steps"] != 600:
        raise ValueError("P1 requires the predeclared 600-step T1 checkpoints")
    annotation = Path(manifest["args"]["dataset_root"]) / "Strawberry/annotations/instances_val2017.json"
    evaluation = Path(manifest["args"]["dataset_root"]) / ".cache_fastdd/sdd_val.clean.json"
    if sha256(annotation) != t1["annotation_sha256"]["val"] or sha256(evaluation) != manifest["annotation_sha256"]:
        raise ValueError("Raw or evaluator annotation hash mismatch")
    data = json.loads(evaluation.read_text())
    audit_annotations(json.loads(annotation.read_text()), data)
    if set(manifest["image_ids"]) != {r["id"] for r in data["images"]}:
        raise ValueError("Evaluation image IDs differ from full annotation")
    anns = [r for r in data["annotations"] if not r.get("iscrowd", 0)]
    image_counts = Counter(r["image_id"] for r in anns)
    category_counts = Counter(r["category_id"] for r in anns)
    names = {r["id"]: r["name"] for r in data["categories"]}
    counts = dict(all=len(anns), crowded_ge5=sum(n for n in image_counts.values() if n >= 5))
    for category, n in category_counts.items():
        counts[f"category_{manifest['category_names'].index(names[category])}"] = n
    cfg = yaml.safe_load((root / "config.yaml").read_text())
    dd = cfg["MODEL"]["DiffusionDet"]
    required = dict(NUM_HEADS=4, EVAL_NUM_HEADS=0, NUM_PROPOSALS=300, INFERENCE_TOPK=300,
                    SAMPLE_STEP=1, FORMULATION="vp", SOLVER="ddim", BOX_RENEWAL=False,
                    USE_ENSEMBLE=False, ATTENTION_NEED_WEIGHTS=False, USE_NMS=True, DEEP_SUPERVISION=True)
    if any(dd[k] != v for k, v in required.items()):
        raise ValueError("P1 config does not match the frozen protocol")
    if set(manifest["checkpoint_sha256"]) != {str(s) for s in manifest["args"]["train_seeds"]}:
        raise ValueError("Missing checkpoint provenance")
    for seed, digest in manifest["checkpoint_sha256"].items():
        arm = checkpoint_root / f"seed{seed}_uniform"
        training = json.loads((arm / "training_summary.json").read_text())
        if digest != training["checkpoint_sha256"] or sha256(arm / "model_final.pth") != digest:
            raise ValueError(f"Checkpoint hash mismatch for seed {seed}")
        old = yaml.safe_load((arm / "config.yaml").read_text())
        for key in ("INPUT", "DATASETS"):
            if cfg[key] != old[key]:
                raise ValueError(f"P1 and T1 {key} settings differ")
    return counts, read_rows(checkpoint_root / "metrics.csv")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", type=Path)
    p.add_argument("--prefix", default="v3_proposal_sweep")
    args = p.parse_args()
    root = args.run_dir
    manifest = json.loads((root / "manifest.json").read_text())
    completed = json.loads((root / "COMPLETE.json").read_text())
    rows, coverage = read_rows(root / "metrics.csv"), read_rows(root / "coverage.csv")
    parity = json.loads((root / "parity.json").read_text())
    counts, reference = audit_files(root, manifest)
    audit, metrics = audit_matrix(rows, coverage, manifest, completed, parity, reference)
    if audit["gt_counts"] != counts:
        raise ValueError("Coverage GT denominators differ from annotation")
    paired, summary, per_class, gates = summarize(rows, coverage, manifest, metrics)
    report = dict(audit=audit, candidates=gates, report_script_sha256=sha256(__file__),
                  evidence_sha256={name: sha256(root / name) for name in
                                   ("metrics.csv", "coverage.csv", "parity.json", "manifest.json", "config.yaml", "COMPLETE.json")},
                  aggregation="Mean inference seeds within checkpoint, then equal weight across checkpoints; p50 reduction averages checkpoint ratios",
                  scope="SDD development screen of frozen H4 checkpoints, not small-P retraining or G4 confirmation")
    raw = Path("results/raw")
    raw.mkdir(parents=True, exist_ok=True)
    for name, suffix in (("metrics.csv", ".csv"), ("coverage.csv", "_coverage.csv"),
                         ("manifest.json", "_manifest.json"), ("config.yaml", "_config.yaml"),
                         ("parity.json", "_parity.json")):
        shutil.copyfile(root / name, raw / f"{args.prefix}{suffix}")
    for suffix, values in (("paired", paired), ("summary", summary), ("per_class", per_class)):
        write_rows(raw / f"{args.prefix}_{suffix}.csv", values)
    (raw / f"{args.prefix}_gate.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = ["| P | AP | ΔAP | AP75 | ΔAP75 | p50 ms | p95 ms | p50 reduction | Crowded Δcoverage50 pp | Gate |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for r in summary:
        verdict = "baseline" if r["proposals"] == 300 else str(next(g for g in gates if g["proposals"] == r["proposals"])["passed_development_screen"])
        lines.append(f"| {r['proposals']} | {r['AP']:.3f} | {r['delta_AP']:+.3f} | {r['AP75']:.3f} | {r['delta_AP75']:+.3f} | {r['latency_p50_ms']:.3f} | {r['latency_p95_ms']:.3f} | {r['p50_reduction_percent']:.2f}% | {r['delta_crowded_ge5_coverage50_pp']:+.3f} | {verdict} |")
    (root / "summary.md").write_text("\n".join(lines) + "\n\n" + json.dumps(report, indent=2) + "\n")
    print("\n".join(lines))
    print(json.dumps(report, indent=2))
    os.environ.setdefault("MPLCONFIGDIR", str(root.resolve() / "matplotlib_cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))
    for seed in manifest["args"]["train_seeds"]:
        selected = sorted([r for r in paired if r["train_seed"] == seed], key=lambda r: r["proposals"])
        axes[0].plot([r["latency_p50_ms"] for r in selected], [r["AP"] for r in selected],
                     marker="o", alpha=0.6, label=f"Checkpoint seed {seed}")
    axes[0].plot([r["latency_p50_ms"] for r in summary], [r["AP"] for r in summary],
                 color="black", marker="D", linestyle=":", label="Mean (P labels)")
    for row in summary:
        axes[0].annotate(f"P{row['proposals']}", (row["latency_p50_ms"], row["AP"]), xytext=(4, 4), textcoords="offset points")
    axes[0].set(xlabel="Model call p50 (ms)", ylabel="SDD AP")
    axes[0].legend(fontsize=8)
    for metric, label in (("delta_AP", "AP"), ("delta_AP75", "AP75")):
        axes[1].plot([r["proposals"] for r in summary], [r[metric] for r in summary], marker="o", label=label)
    axes[1].axhline(-0.3, color="gray", linestyle="--", label="Mean floor -0.3")
    axes[1].set(xlabel="Proposals per stage", ylabel="Paired AP difference")
    axes[1].legend()
    for count in manifest["args"]["proposals"]:
        values = [100 * average([r for r in coverage if int(r["proposals"]) == count
                                and int(r["stage"]) == stage and r["group"] == "crowded_ge5"], "coverage50")
                  for stage in range(1, 5)]
        axes[2].plot(range(1, 5), values, marker="o", label=f"P{count}")
    axes[2].set(xticks=range(1, 5), xlabel="Head stage", ylabel="Crowded GT coverage @ IoU .5 (%)")
    axes[2].legend()
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.suptitle("Frozen H4 checkpoints; shared noise pool; fixed pre-NMS top-k 300\n3 checkpoint seeds × 3 inference seeds × 750 images; coverage is not COCO AR")
    fig.tight_layout()
    Path("results/figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(f"results/figures/{args.prefix}.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
