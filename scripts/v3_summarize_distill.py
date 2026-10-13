"""Summarize paired training runs without treating inference seeds as training repeats."""
import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import statistics as stats


def read_rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def write_rows(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", type=Path)
    p.add_argument("--prefix", default="v3_kd_pilot")
    args = p.parse_args()
    root = args.run_dir
    manifest = json.loads((root / "manifest.json").read_text())
    if not (root / "COMPLETE.json").is_file():
        raise ValueError("Run is not complete")
    rows = read_rows(root / "metrics.csv")
    seeds = manifest["args"]["train_seeds"]
    eval_seeds = manifest["args"]["eval_seeds"]
    expected = {(s, a, e) for s in seeds for a in ("control", "kd") for e in eval_seeds}
    actual = {(int(r["train_seed"]), r["arm"], int(r["eval_seed"])) for r in rows}
    if actual != expected or len(rows) != len(expected) or not all(r["full_dataset"] == "True" for r in rows):
        raise ValueError("Need complete paired arms and full-validation rows")
    average = lambda rs, key: stats.mean(float(r[key]) for r in rs)
    paired, audits = [], []
    for seed in seeds:
        groups = {a: [r for r in rows if int(r["train_seed"]) == seed and r["arm"] == a] for a in ("control", "kd")}
        info = {a: json.loads((root / f"seed{seed}_{a}" / "training_summary.json").read_text()) for a in groups}
        for key in ("initial_state_sha256", "batch_trace_sha256", "image_passes"):
            if info["control"][key] != info["kd"][key]:
                raise ValueError(f"Unpaired {key} for seed {seed}")
        if not info["kd"]["teacher_unchanged"]:
            raise ValueError("Teacher freeze check failed")
        paired.append(dict(train_seed=seed, control_AP=average(groups["control"], "AP"),
                           kd_AP=average(groups["kd"], "AP"),
                           delta_AP=average(groups["kd"], "AP")-average(groups["control"], "AP"),
                           delta_AP50=average(groups["kd"], "AP50")-average(groups["control"], "AP50"),
                           delta_AP75=average(groups["kd"], "AP75")-average(groups["control"], "AP75"),
                           control_p50_ms=average(groups["control"], "latency_p50_ms"),
                           kd_p50_ms=average(groups["kd"], "latency_p50_ms")))
        for arm in groups:
            logs = [json.loads(line) for line in (root / f"seed{seed}_{arm}" / "training.jsonl").read_text().splitlines()]
            if len(logs) != manifest["args"]["steps"]:
                raise ValueError("Incomplete training log")
            accepted = sum(r.get("kd_pairs", 0) for r in logs)
            positives = sum(r.get("student_positives", 0) for r in logs)
            audits.append(dict(train_seed=seed, arm=arm, **info[arm],
                               mean_kd_pairs=accepted/len(logs), accepted_fraction=accepted/max(1, positives),
                               kd_teacher_iou=sum(r.get("kd_pairs",0)*r.get("kd_teacher_iou",0) for r in logs)/max(1,accepted),
                               kd_student_iou=sum(r.get("kd_pairs",0)*r.get("kd_student_iou",0) for r in logs)/max(1,accepted),
                               final20_supervised_loss=stats.mean(r["loss_total"]-r.get("loss_kd_cls",0)-r.get("loss_kd_box",0) for r in logs[-20:]),
                               final20_kd_cls=stats.mean(r.get("loss_kd_cls",0) for r in logs[-20:]),
                               final20_kd_box=stats.mean(r.get("loss_kd_box",0) for r in logs[-20:])))
    raw = Path("results/raw")
    raw.mkdir(parents=True, exist_ok=True)
    write_rows(raw / f"{args.prefix}_paired.csv", paired)
    write_rows(raw / f"{args.prefix}_training.csv", audits)
    shutil.copyfile(root / "metrics.csv", raw / f"{args.prefix}.csv")
    shutil.copyfile(root / "manifest.json", raw / f"{args.prefix}_manifest.json")
    shutil.copyfile(root / "config.yaml", raw / f"{args.prefix}_config.yaml")
    category_rows = []
    for key in rows[0]:
        if key.startswith("AP-"):
            c = average([r for r in rows if r["arm"] == "control"], key)
            k = average([r for r in rows if r["arm"] == "kd"], key)
            category_rows.append(dict(category=key[3:], control_AP=c, kd_AP=k, delta_AP=k-c))
    write_rows(raw / f"{args.prefix}_per_class.csv", category_rows)
    lines = ["| Training seed | Control AP | KD AP | Paired ΔAP | ΔAP50 | ΔAP75 |",
             "|---|---|---|---|---|---|"]
    for r in paired:
        lines.append(f"| {r['train_seed']} | {r['control_AP']:.3f} | {r['kd_AP']:.3f} | {r['delta_AP']:+.3f} | {r['delta_AP50']:+.3f} | {r['delta_AP75']:+.3f} |")
    lines.append(f"| Mean | {average(paired,'control_AP'):.3f} | {average(paired,'kd_AP'):.3f} | {average(paired,'delta_AP'):+.3f} | {average(paired,'delta_AP50'):+.3f} | {average(paired,'delta_AP75'):+.3f} |")
    (root / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("Paired training-seed delta sample SD:", stats.stdev(r["delta_AP"] for r in paired) if len(paired)>1 else "undefined")

    os.environ.setdefault("MPLCONFIGDIR", str(root.resolve() / "matplotlib_cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for r in paired:
        ax.plot([0, 1], [r["control_AP"], r["kd_AP"]], marker="o", label=f"Training seed {r['train_seed']}")
    ax.set(xticks=[0, 1], xticklabels=["4-stage control", "4-stage quality-gated KD"], ylabel="SDD AP",
           title=f"Paired {manifest['args']['steps']}-step fine-tuning\nEach point averages {len(eval_seeds)} inference seeds on 750 images")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    Path("results/figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(f"results/figures/{args.prefix}.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
