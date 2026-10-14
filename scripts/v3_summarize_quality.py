"""Audit Q1 against its reused controls, then report quality and depth effects separately."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics as stats
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.v3_head_sweep import sha256
from scripts.v3_quality import ARMS, check_config
from scripts.v3_summarize_distill import read_rows, write_rows
from scripts.v3_summarize_time_alignment import audit_log


def average(rows, key):
    return stats.mean(float(r[key]) for r in rows)


def audit_training(logs, reference, info, initial_h4_digest):
    if len(logs) != 600 or len(reference) != 600 or info["executed_steps"] != 600:
        raise ValueError("Q1 requires complete 600-step training")
    computed = audit_log(logs, "uniform", 0.5, 8)
    for key, value in computed.items():
        if info[key] != value:
            raise ValueError(f"Training summary mismatch: {key}")
    for row, original in zip(logs, reference):
        if row["pair_digest"] != original["pair_digest"] or row["lr"] != original["lr"] or row["timesteps"] != original["timesteps"]:
            raise ValueError("Training input/RNG/timestep/LR pairing differs from T1")
        if any(k.startswith("loss_kd") for k in row):
            raise ValueError("Q1 must not use distillation")
    if info.get("initial_h4_prefix_sha256", info["initial_state_sha256"]) != initial_h4_digest:
        raise ValueError("H4 prefix initialization differs from T1")
    if info["final_class_loss"] == "quality_focal":
        if (not all(r["quality_positives"] >= 0 and 0 <= r["quality_mean_iou"] <= 1 for r in logs)
                or sum(r["quality_positives"] for r in logs) == 0):
            raise ValueError("Missing or invalid quality supervision diagnostics")
    elif any("quality_positives" in r for r in logs):
        raise ValueError("Focal control unexpectedly used quality supervision")
    return computed


def audit_matrix(rows, manifest, completed, reference):
    args = manifest["args"]
    seeds, eval_seeds = args["train_seeds"], args["eval_seeds"]
    if (set(seeds) != {10, 11, 12} or len(seeds) != 3 or set(eval_seeds) != {0, 1, 2}
            or len(eval_seeds) != 3 or args["steps"] != 600 or args["eval_limit"] != 0):
        raise ValueError("Q1 final report requires the complete predeclared protocol")
    expected = {(s, a, e) for s in seeds for a in ARMS for e in eval_seeds}
    key = lambda r: (int(r["train_seed"]), r["arm"], int(r["eval_seed"]))
    if ({key(r) for r in rows} != expected or len(rows) != 27
            or completed != dict(evaluations=27, new_training_runs=6, reused_controls=3)):
        raise ValueError("Incomplete or duplicate evaluation matrix")
    if any(str(r["full_dataset"]) != "True" or int(r["steps"]) != 600
           or int(r["images"]) != manifest["full_val_images"] or int(r["measured_images"]) != 200 for r in rows):
        raise ValueError("Need full validation and standard timing")
    if len(manifest["image_ids"]) != manifest["full_val_images"] or len(set(manifest["image_ids"])) != manifest["full_val_images"]:
        raise ValueError("Incomplete validation image set")
    metrics = [k for k in rows[0] if k.startswith("AP") or k.startswith("latency_")]
    if any(not math.isfinite(float(r[k])) for r in rows for k in metrics):
        raise ValueError("Non-finite metrics")
    if any(float(r["latency_p50_ms"]) <= 0 or float(r["latency_p95_ms"]) < float(r["latency_p50_ms"]) for r in rows):
        raise ValueError("Invalid timing quantiles")
    reference = [r for r in reference if r["arm"] == "uniform"]
    ref = {(int(r["train_seed"]), int(r["eval_seed"])): r for r in reference}
    if len(ref) != 9 or len(reference) != 9:
        raise ValueError("Incomplete reference evaluation matrix")
    if {k for k in metrics if k.startswith("AP")} != {k for k in reference[0] if k.startswith("AP")}:
        raise ValueError("Incomplete AP or category metrics")
    discrepancies = []
    for row in rows:
        if row["arm"] != "h4_focal":
            continue
        original = ref[(int(row["train_seed"]), int(row["eval_seed"]))]
        delta = [float(row[k]) - float(original[k]) for k in metrics if k.startswith("AP")]
        if any(not math.isfinite(d) or abs(d) > 1e-5 for d in delta):
            raise ValueError("Reevaluated control AP differs from T1")
        discrepancies.extend(abs(d) for d in delta)
    return metrics, max(discrepancies)


def summarize(rows, metrics):
    paired = []
    for seed in (10, 11, 12):
        groups = {arm: [r for r in rows if int(r["train_seed"]) == seed and r["arm"] == arm] for arm in ARMS}
        row = dict(train_seed=seed)
        for key in metrics:
            for arm in ARMS:
                row[f"{arm}_{key}"] = average(groups[arm], key)
            row[f"quality_delta_{key}"] = row[f"h4_quality_{key}"] - row[f"h4_focal_{key}"]
            row[f"depth_delta_{key}"] = row[f"h4_focal_{key}"] - row[f"h6_focal_{key}"]
        row["quality_p50_change_percent"] = 100 * (row["h4_quality_latency_p50_ms"] / row["h4_focal_latency_p50_ms"] - 1)
        row["depth_p50_reduction_percent"] = 100 * (1 - row["h4_focal_latency_p50_ms"] / row["h6_focal_latency_p50_ms"])
        paired.append(row)
    categories = [dict(category=k[3:], **{f"{arm}_AP": average(paired, f"{arm}_{k}") for arm in ARMS},
                       quality_delta_AP=average(paired, f"quality_delta_{k}"), depth_delta_AP=average(paired, f"depth_delta_{k}"))
                  for k in metrics if k.startswith("AP-")]
    worst = min(categories, key=lambda r: r["quality_delta_AP"])
    gate = dict(mean_delta_AP=average(paired, "quality_delta_AP"), mean_delta_AP75=average(paired, "quality_delta_AP75"),
                positive_pairs=sum(r["quality_delta_AP"] > 0 for r in paired),
                delta_AP_sample_sd=stats.stdev(r["quality_delta_AP"] for r in paired),
                p50_change_percent=average(paired, "quality_p50_change_percent"),
                worst_category=worst["category"], worst_category_delta_AP=worst["quality_delta_AP"])
    gate["checks"] = dict(AP=gate["mean_delta_AP"] >= 0.5, AP75=gate["mean_delta_AP75"] > 0,
                          positive_pairs=gate["positive_pairs"] >= 2, latency=gate["p50_change_percent"] <= 5,
                          worst_category=gate["worst_category_delta_AP"] >= -2)
    gate["passed_development_screen"] = all(gate["checks"].values())
    depth = dict(mean_H4_minus_H6_AP=average(paired, "depth_delta_AP"),
                 mean_H4_minus_H6_AP75=average(paired, "depth_delta_AP75"),
                 H4_p50_reduction_percent=average(paired, "depth_p50_reduction_percent"),
                 scope="Equal 600-step fine-tuning; includes depth and auxiliary supervision changes; not the Q1 loss effect")
    return paired, categories, gate, depth


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--prefix", default="v3_quality")
    args = parser.parse_args()
    root = args.run_dir
    manifest = json.loads((root / "manifest.json").read_text())
    completed = json.loads((root / "COMPLETE.json").read_text())
    rows = read_rows(root / "metrics.csv")
    ref_root = Path(manifest["args"]["reference_root"])
    if sha256(ref_root / "manifest.json") != manifest["reference_manifest_sha256"]:
        raise ValueError("T1 reference manifest hash mismatch")
    reference_manifest = json.loads((ref_root / "manifest.json").read_text())
    if any(manifest[k] != reference_manifest[k] for k in ("torch", "detectron2", "cuda", "gpu")):
        raise ValueError("Q1 environment differs from the reused T1 control")
    for source, digest in manifest["source_sha256"].items():
        if sha256(root / "source_snapshot" / source) != digest:
            raise ValueError(f"Source snapshot hash mismatch: {source}")
    for source, digest in manifest["reference_snapshot_sha256"].items():
        if sha256(root / source) != digest:
            raise ValueError(f"Reference snapshot hash mismatch: {source}")
    if sha256(manifest["args"]["weights"]) != manifest["initialization_sha256"]:
        raise ValueError("Original initialization checkpoint hash mismatch")
    for split, digest in manifest["annotation_sha256"].items():
        if sha256(Path(manifest["args"]["dataset_root"]) / f"Strawberry/annotations/instances_{split}2017.json") != digest:
            raise ValueError("Annotation hash mismatch")
    metrics, max_control_delta = audit_matrix(rows, manifest, completed, read_rows(ref_root / "metrics.csv"))
    import yaml
    replays = json.loads((root / "control_replay.json").read_text())
    if len(replays) != 3 or {r["train_seed"] for r in replays} != {10, 11, 12}:
        raise ValueError("Missing control loss replays")
    training = []
    for seed in manifest["args"]["train_seeds"]:
        ref_dir = root / "reference_snapshot" / f"seed{seed}_uniform"
        ref_info = json.loads((ref_dir / "training_summary.json").read_text())
        ref_logs = [json.loads(s) for s in (ref_dir / "training.jsonl").read_text().splitlines()]
        replay = next(r for r in replays if r["train_seed"] == seed)
        if (replay["initial_state_sha256"] != ref_info["initial_state_sha256"]
                or replay["pair_digest"] != ref_logs[0]["pair_digest"]):
            raise ValueError("Control replay is unpaired")
        expected_losses = {k for k in ref_logs[0] if k.startswith("loss_") and k != "loss_total"}
        if (set(replay["loss_differences"]) != expected_losses
                or any(not math.isfinite(d) or abs(d) > max(1e-6, abs(ref_logs[0][k])*1e-6)
                       for k, d in replay["loss_differences"].items())):
            raise ValueError("Control loss replay differs from T1")
        for arm in ARMS:
            arm_dir = root / f"seed{seed}_{arm}"
            info = json.loads((arm_dir / "training_summary.json").read_text())
            if sha256(info["checkpoint_path"]) != info["checkpoint_sha256"]:
                raise ValueError("Checkpoint hash mismatch")
            heads = 6 if arm == "h6_focal" else 4
            mode = "quality_focal" if arm == "h4_quality" else "focal"
            if (info["heads"] != heads or info["final_class_loss"] != mode
                    or info["trained_this_run"] != (arm != "h4_focal")):
                raise ValueError("Arm structure or loss differs from protocol")
            if arm != "h6_focal" and info["parameters"] != ref_info["parameters"]:
                raise ValueError("H4 parameter count changed")
            if arm == "h4_focal" and info["checkpoint_sha256"] != manifest["reference_checkpoint_sha256"][str(seed)]:
                raise ValueError("Wrong reused control")
            cfg = yaml.safe_load((arm_dir / "config.yaml").read_text())
            if cfg["SOLVER"]["WARMUP_FACTOR"] != 0.1:
                raise ValueError("Q1 configuration must record the actual warmup factor")
            if cfg["MODEL"]["DiffusionDet"]["NUM_HEADS"] != heads or cfg["MODEL"]["DiffusionDet"]["FINAL_CLASS_LOSS"] != mode:
                raise ValueError("Incorrect arm configuration")
            cfg["MODEL"]["DiffusionDet"]["NUM_HEADS"] = 4
            cfg["MODEL"]["DiffusionDet"]["FINAL_CLASS_LOSS"] = "focal"
            check_config(cfg, yaml.safe_load((ref_dir / "config.yaml").read_text()))
            logs = ref_logs if arm == "h4_focal" else [json.loads(s) for s in (arm_dir / "training.jsonl").read_text().splitlines()]
            audit_training(logs, ref_logs, info, ref_info["initial_state_sha256"])
            training.append(dict(train_seed=seed, arm=arm, **{k: info[k] for k in
                                 ("trained_this_run", "heads", "parameters", "train_wall_s", "peak_train_allocated_gb",
                                  "checkpoint_sha256", "checkpoint_path", "initial_state_sha256", "pairing_trace_sha256", "image_passes")},
                                 final20_loss=average(logs[-20:], "loss_total"),
                                 final20_quality_iou=average(logs[-20:], "quality_mean_iou") if arm == "h4_quality" else None))
    paired, categories, gate, depth = summarize(rows, metrics)
    report = dict(quality_gate=gate, matched_depth_comparison=depth, control_max_abs_AP_difference=max_control_delta,
                  training_pairs=3, new_training_runs=6, evaluation_rows=27,
                  report_script_sha256=sha256(__file__),
                  evidence_sha256={name: sha256(root / name) for name in ("metrics.csv", "manifest.json", "control_replay.json", "COMPLETE.json")},
                  scope="SDD development comparison; inference seeds averaged within each model; independent confirmation still required")
    raw = Path("results/raw")
    raw.mkdir(parents=True, exist_ok=True)
    for name, suffix in (("metrics.csv", ".csv"), ("manifest.json", "_manifest.json"),
                         ("config.yaml", "_config.yaml"), ("control_replay.json", "_control_replay.json")):
        shutil.copyfile(root / name, raw / f"{args.prefix}{suffix}")
    for name, values in (("paired", paired), ("per_class", categories), ("training", training)):
        write_rows(raw / f"{args.prefix}_{name}.csv", values)
    (raw / f"{args.prefix}_gate.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = ["| Training seed | H4 focal AP | H4 quality AP | Q1 ΔAP | Q1 ΔAP75 | H6 focal AP | H4−H6 AP |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    mean = {k: average(paired, k) for k in paired[0]}
    for index, row in enumerate([*paired, mean]):
        seed = row["train_seed"] if index < len(paired) else "Mean"
        lines.append(f"| {seed} | {row['h4_focal_AP']:.3f} | {row['h4_quality_AP']:.3f} | {row['quality_delta_AP']:+.3f} | {row['quality_delta_AP75']:+.3f} | {row['h6_focal_AP']:.3f} | {row['depth_delta_AP']:+.3f} |")
    (root / "summary.md").write_text("\n".join(lines) + "\n\n" + json.dumps(report, indent=2) + "\n")
    print("\n".join(lines))
    print(json.dumps(report, indent=2))
    os.environ.setdefault("MPLCONFIGDIR", str(root.resolve() / "matplotlib_cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))
    for row in paired:
        axes[0].plot([0, 1], [row["h4_focal_AP"], row["h4_quality_AP"]], marker="o", label=f"Train seed {row['train_seed']}")
        axes[1].scatter(row["train_seed"], row["quality_delta_AP"], marker="o", color="tab:blue")
        axes[1].scatter(row["train_seed"], row["quality_delta_AP75"], marker="x", color="tab:orange")
    axes[0].set(xticks=[0, 1], xticklabels=["H4 focal", "H4 quality focal"], ylabel="SDD AP")
    axes[0].legend()
    axes[1].scatter([], [], marker="o", color="tab:blue", label="ΔAP")
    axes[1].scatter([], [], marker="x", color="tab:orange", label="ΔAP75")
    axes[1].axhline(0.5, color="gray", linestyle="--", label="Mean AP gate +0.5")
    axes[1].axhline(0, color="gray", linewidth=1)
    axes[1].set(xticks=[10, 11, 12], xlabel="Training seed", ylabel="Quality − focal AP")
    axes[1].legend(fontsize=8)
    for arm in ARMS:
        axes[2].scatter([r[f"{arm}_latency_p50_ms"] for r in paired], [r[f"{arm}_AP"] for r in paired], label=arm)
    axes[2].set(xlabel="Model call p50 (ms)", ylabel="SDD AP")
    axes[2].legend()
    for ax in axes:
        ax.grid(alpha=0.2)
    verdict = "passed" if gate["passed_development_screen"] else "not passed"
    fig.suptitle(f"Q1: final-head quality supervision; development gate {verdict}\n600 steps per model; 3 train seeds × 3 inference seeds × 750 images; P300, uniform t")
    fig.tight_layout()
    Path("results/figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(f"results/figures/{args.prefix}.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
