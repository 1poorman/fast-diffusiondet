"""Archive a completed sweep and reproduce its table and Pareto figure."""
import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import statistics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", type=Path)
    p.add_argument("--prefix", default="v3_head_sweep")
    args = p.parse_args()
    manifest = json.loads((args.run_dir / "manifest.json").read_text())
    rows = list(csv.DictReader((args.run_dir / "metrics.csv").open()))
    expected = {(seed, int(arm.split(":")[0]), arm.split(":")[1])
                for seed in manifest["args"]["seeds"] for arm in manifest["args"]["arms"]}
    actual = {(int(r["seed"]), int(r["depth"]), r["attention"]) for r in rows}
    if len(actual) != len(rows) or actual != expected or not all(r["full_dataset"] == "True" for r in rows):
        raise ValueError("Refusing to summarize incomplete/duplicate/subset results as a full sweep")
    groups = {}
    for row in rows:
        groups.setdefault((int(row["depth"]), row["attention"]), []).append(row)
    baseline_depth = max(depth for depth, _ in groups)
    baseline = {int(r["seed"]): r for r in groups[(baseline_depth, "legacy")]}
    mean = lambda group, field: statistics.mean(float(r[field]) for r in group)
    summary = []
    for (depth, attention), group in groups.items():
        deltas = [float(r["AP"]) - float(baseline[int(r["seed"])]["AP"]) for r in group]
        ratios = [float(baseline[int(r["seed"])]["latency_p50_ms"]) / float(r["latency_p50_ms"]) for r in group]
        summary.append(dict(depth=depth, attention=attention, seeds=len(group),
                            AP=mean(group, "AP"), AP_std=statistics.stdev(float(r["AP"]) for r in group) if len(group)>1 else 0,
                            AP50=mean(group, "AP50"), AP75=mean(group, "AP75"),
                            delta_AP=statistics.mean(deltas), delta_AP_min=min(deltas), delta_AP_max=max(deltas),
                            p50_ms=mean(group, "latency_p50_ms"), p95_ms=mean(group, "latency_p95_ms"),
                            paired_speedup=statistics.mean(ratios)))
    raw = Path("results/raw")
    raw.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.run_dir / "metrics.csv", raw / f"{args.prefix}.csv")
    shutil.copyfile(args.run_dir / "manifest.json", raw / f"{args.prefix}_manifest.json")
    with (raw / f"{args.prefix}_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    per_class = []
    for (depth, attention), group in groups.items():
        metrics = [json.loads((args.run_dir / f"seed{r['seed']}_d{depth}_{attention}" / "bbox_metrics.json").read_text()) for r in group]
        for name in metrics[0]:
            if name.startswith("AP-"):
                per_class.append(dict(depth=depth, attention=attention, category=name[3:], AP=mean(metrics, name)))
    with (raw / f"{args.prefix}_per_class.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["depth", "attention", "category", "AP"])
        w.writeheader()
        w.writerows(per_class)
    lines = ["| H | Attention | AP (mean±sample std) | ΔAP paired | AP75 | p50 ms | p95 ms | Speedup |",
             "|---|---|---|---|---|---|---|---|"]
    for r in summary:
        lines.append(f"| {r['depth']} | {r['attention']} | {r['AP']:.3f}±{r['AP_std']:.3f} | {r['delta_AP']:+.3f} | {r['AP75']:.3f} | {r['p50_ms']:.2f} | {r['p95_ms']:.2f} | {r['paired_speedup']:.3f}× |")
    (args.run_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    os.environ.setdefault("MPLCONFIGDIR", str(args.run_dir.resolve() / "matplotlib_cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    for r in summary:
        ax.errorbar(r["p50_ms"], r["AP"], yerr=r["AP_std"], fmt="o", capsize=4,
                    label=f"H={r['depth']} / {r['attention']}")
    seed_count = len(manifest["args"]["seeds"])
    ax.set(xlabel="Batch-1 model latency, mean of per-seed p50 (ms)", ylabel="SDD validation AP",
           title=f"Frozen checkpoint, NFE=1, P=300, FP32\n{seed_count} evaluation seeds; error bars are sampling SD, not training uncertainty")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)
    fig.tight_layout()
    figures = Path("results/figures")
    figures.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures / f"{args.prefix}.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
