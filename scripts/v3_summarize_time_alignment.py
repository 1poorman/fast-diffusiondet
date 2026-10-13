"""Validate and archive T1 paired trials; inference seeds are averaged within model."""
import argparse
from collections import Counter
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
from scripts.v3_summarize_distill import read_rows, write_rows


def audit_log(logs, arm, probability, batch_size):
    counts, pair_trace, inputs = Counter(), [], []
    for step, row in enumerate(logs):
        if row["step"] != step or not math.isfinite(row["loss_total"]):
            raise ValueError("Non-sequential or non-finite training log")
        if not all(len(row[k]) == batch_size for k in ("timesteps", "base_timesteps", "gates")):
            raise ValueError("Wrong number of timestep records")
        for base, gate, actual in zip(row["base_timesteps"], row["gates"], row["timesteps"]):
            expected = 999 if arm == "mix_tmax" and gate < probability else base
            if not 0 <= base < 1000 or not 0 <= gate < 1 or actual != expected:
                raise ValueError("Timestep does not follow configured distribution")
        pair_record = {k: row[k] for k in ("input_digest", "source_digest", "base_timesteps",
                                          "gates", "rng_after_forward_digest")}
        digest = hashlib.sha256(json.dumps(pair_record, sort_keys=True).encode()).hexdigest()
        if digest != row["pair_digest"]:
            raise ValueError("Pair digest does not match recorded sources")
        pair_trace.append(digest)
        inputs.append(row["input_digest"])
        counts.update(row["timesteps"])
    return dict(pairing_trace_sha256=hashlib.sha256("".join(pair_trace).encode()).hexdigest(),
                batch_trace_sha256=hashlib.sha256("".join(inputs).encode()).hexdigest(),
                timestep_histogram={str(t): n for t, n in sorted(counts.items())},
                image_passes=sum(counts.values()), tmax_fraction=counts[999]/sum(counts.values()),
                timestep_mean=sum(t*n for t,n in counts.items())/sum(counts.values()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", type=Path)
    p.add_argument("--prefix", default="v3_time_alignment")
    args = p.parse_args()
    root = args.run_dir
    manifest = json.loads((root / "manifest.json").read_text())
    if not (root / "COMPLETE.json").is_file():
        raise ValueError("Run is not complete")
    completed = json.loads((root / "COMPLETE.json").read_text())
    rows = read_rows(root / "metrics.csv")
    seeds, eval_seeds = manifest["args"]["train_seeds"], manifest["args"]["eval_seeds"]
    expected = {(s, a, e) for s in seeds for a in ("uniform", "mix_tmax") for e in eval_seeds}
    actual = {(int(r["train_seed"]), r["arm"], int(r["eval_seed"])) for r in rows}
    if (actual != expected or len(rows) != len(expected) or completed["evaluation_rows"] != len(rows)
            or completed["runs"] != len(seeds)*2):
        raise ValueError("Incomplete or duplicate paired evaluation matrix")
    if any(r["full_dataset"] != "True" or int(r["images"]) != manifest["full_val_images"]
           or int(r["steps"]) != manifest["args"]["steps"] or int(r["measured_images"]) != 200 for r in rows):
        raise ValueError("Need full validation and the standard timing protocol")
    metric_keys = [k for k in rows[0] if k.startswith("AP") or k.startswith("latency_")]
    if any(not math.isfinite(float(r[k])) for r in rows for k in metric_keys):
        raise ValueError("Non-finite validation metrics")
    for source, digest in manifest["source_sha256"].items():
        if sha256(root / "source_snapshot" / source) != digest:
            raise ValueError(f"Source snapshot differs: {source}")
    average = lambda rs, key: stats.mean(float(r[key]) for r in rs)
    paired, audits = [], []
    import yaml
    base_config = yaml.safe_load((root / "config.yaml").read_text())
    batch_size = base_config["SOLVER"]["IMS_PER_BATCH"]
    for seed in seeds:
        groups = {a: [r for r in rows if int(r["train_seed"]) == seed and r["arm"] == a]
                  for a in ("uniform", "mix_tmax")}
        info = {a: json.loads((root / f"seed{seed}_{a}" / "training_summary.json").read_text()) for a in groups}
        for key in ("initial_state_sha256", "batch_trace_sha256", "pairing_trace_sha256", "image_passes", "parameters"):
            if info["uniform"][key] != info["mix_tmax"][key]:
                raise ValueError(f"Unpaired {key}, seed {seed}")
        pair = dict(train_seed=seed)
        for key in metric_keys:
            u, m = average(groups["uniform"], key), average(groups["mix_tmax"], key)
            pair.update({f"uniform_{key}": u, f"mix_tmax_{key}": m, f"delta_{key}": m-u})
        paired.append(pair)
        reference_config, reference_lr = None, None
        for arm in groups:
            arm_dir = root / f"seed{seed}_{arm}"
            logs = [json.loads(line) for line in (arm_dir / "training.jsonl").read_text().splitlines()]
            if len(logs) != manifest["args"]["steps"]:
                raise ValueError("Incomplete training log")
            arm_config = yaml.safe_load((arm_dir / "config.yaml").read_text())
            if arm_config["MODEL"]["DiffusionDet"]["TRAIN_T_DIST"] != arm:
                raise ValueError("Arm config has the wrong distribution")
            if arm_config["MODEL"]["DiffusionDet"]["TRAIN_T_HIGH_PROB"] != manifest["args"]["high_probability"]:
                raise ValueError("Arm config has the wrong mixture probability")
            arm_config.pop("OUTPUT_DIR")
            arm_config["MODEL"]["DiffusionDet"].pop("TRAIN_T_DIST")
            learning_rates = [r["lr"] for r in logs]
            if any(k.startswith("loss_kd") for r in logs for k in r):
                raise ValueError("T1 must not contain distillation losses")
            if reference_config is not None and (arm_config != reference_config or learning_rates != reference_lr):
                raise ValueError("Paired settings or learning-rate schedules differ")
            reference_config, reference_lr = arm_config, learning_rates
            if sha256(arm_dir / "model_final.pth") != info[arm]["checkpoint_sha256"]:
                raise ValueError("Checkpoint hash mismatch")
            computed = audit_log(logs, arm, manifest["args"]["high_probability"], batch_size)
            for key, value in computed.items():
                if info[arm][key] != value:
                    raise ValueError(f"Training summary mismatch: {key}")
            if computed["image_passes"] != batch_size*manifest["args"]["steps"]:
                raise ValueError("Incorrect training budget")
            details = {k:v for k,v in info[arm].items() if k != "timestep_histogram"}
            audits.append(dict(train_seed=seed, arm=arm, **details,
                               final20_loss=stats.mean(r["loss_total"] for r in logs[-20:])))
    raw = Path("results/raw")
    raw.mkdir(parents=True, exist_ok=True)
    write_rows(raw / f"{args.prefix}_paired.csv", paired)
    write_rows(raw / f"{args.prefix}_training.csv", audits)
    for name, suffix in (("metrics.csv", ".csv"), ("manifest.json", "_manifest.json"), ("config.yaml", "_config.yaml")):
        shutil.copyfile(root / name, raw / f"{args.prefix}{suffix}")
    categories = [dict(category=k[3:], uniform_AP=average(paired, f"uniform_{k}"),
                       mix_tmax_AP=average(paired, f"mix_tmax_{k}"), delta_AP=average(paired, f"delta_{k}"))
                  for k in metric_keys if k.startswith("AP-")]
    write_rows(raw / f"{args.prefix}_per_class.csv", categories)
    gate = dict(paired_mean_delta_AP=average(paired, "delta_AP"),
                paired_mean_delta_AP75=average(paired, "delta_AP75"),
                positive_pairs=sum(r["delta_AP"] > 0 for r in paired), training_pairs=len(paired),
                paired_delta_sample_sd=stats.stdev(r["delta_AP"] for r in paired) if len(paired)>1 else None,
                p50_change_percent=100*(average(paired,"mix_tmax_latency_p50_ms")/average(paired,"uniform_latency_p50_ms")-1))
    gate["checks"] = dict(minimum_training_pairs=len(paired) >= 3,
                           mean_AP=gate["paired_mean_delta_AP"] >= 0.3,
                           positive_pairs=gate["positive_pairs"] / len(paired) >= 2/3,
                           AP75=gate["paired_mean_delta_AP75"] >= -0.1,
                           latency=gate["p50_change_percent"] <= 3)
    gate["passed_development_screen"] = all(gate["checks"].values())
    gate["report_script_sha256"] = sha256(__file__)
    gate["scope"] = "SDD development screen; requires independent data and matched H6 training for architecture claims"
    (raw / f"{args.prefix}_gate.json").write_text(json.dumps(gate, indent=2) + "\n")
    lines = ["| Training seed | Uniform AP | Mix tmax AP | Paired ΔAP | ΔAP75 |",
             "|---|---|---|---|---|"]
    for r in paired:
        lines.append(f"| {r['train_seed']} | {r['uniform_AP']:.3f} | {r['mix_tmax_AP']:.3f} | {r['delta_AP']:+.3f} | {r['delta_AP75']:+.3f} |")
    lines.append(f"| Mean | {average(paired,'uniform_AP'):.3f} | {average(paired,'mix_tmax_AP'):.3f} | {gate['paired_mean_delta_AP']:+.3f} | {gate['paired_mean_delta_AP75']:+.3f} |")
    (root / "summary.md").write_text("\n".join(lines) + "\n\n" + json.dumps(gate, indent=2) + "\n")
    print("\n".join(lines))
    print(json.dumps(gate, indent=2))
    os.environ.setdefault("MPLCONFIGDIR", str(root.resolve() / "matplotlib_cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for r in paired:
        axes[0].plot([0, 1], [r["uniform_AP"], r["mix_tmax_AP"]], marker="o", label=f"Train seed {r['train_seed']}")
        axes[1].scatter(r["train_seed"], r["delta_AP"], s=50)
        axes[2].scatter(r["train_seed"], r["delta_AP75"], s=50)
    axes[0].set(xticks=[0,1], xticklabels=["Uniform t", f"Mix tmax (p={manifest['args']['high_probability']})"], ylabel="SDD AP")
    axes[0].legend()
    axes[1].axhline(0, color="gray", linewidth=1)
    axes[1].axhline(0.3, color="orange", linestyle="--", label="Mean gain target +0.3")
    axes[1].axhline(gate["paired_mean_delta_AP"], color="black", linestyle=":", label="Observed mean")
    axes[1].set(xticks=seeds, xlabel="Training seed", ylabel="Paired AP difference")
    axes[1].legend()
    axes[2].axhline(0, color="gray", linewidth=1)
    axes[2].axhline(-0.1, color="orange", linestyle="--", label="Mean AP75 floor -0.1")
    axes[2].axhline(gate["paired_mean_delta_AP75"], color="black", linestyle=":", label="Observed mean")
    axes[2].set(xticks=seeds, xlabel="Training seed", ylabel="Paired AP75 difference")
    axes[2].legend()
    for ax in axes:
        ax.grid(alpha=0.2)
    verdict = "passed" if gate["passed_development_screen"] else "not passed"
    fig.suptitle(f"Physical H4, KD off, {manifest['args']['steps']} steps per arm; development gate: {verdict}\nEach point averages {len(eval_seeds)} inference seeds on {manifest['full_val_images']} images")
    fig.tight_layout()
    Path("results/figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(f"results/figures/{args.prefix}.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
