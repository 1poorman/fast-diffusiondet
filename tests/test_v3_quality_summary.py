"""Reject invalid reuse of historical controls and enforce the predeclared Q1 gate."""
import copy
import hashlib
import json

import pytest

from scripts.v3_summarize_quality import audit_matrix, audit_training, summarize
from scripts.v3_summarize_time_alignment import audit_log
from scripts.v3_quality import check_config, lr_factor


def evidence():
    manifest = dict(args=dict(train_seeds=[10, 11, 12], eval_seeds=[0, 1, 2], steps=600, eval_limit=0),
                    image_ids=list(range(750)), full_val_images=750)
    rows, reference = [], []
    for seed in (10, 11, 12):
        for e in (0, 1, 2):
            for arm in ("h4_focal", "h4_quality", "h6_focal"):
                gain = dict(h4_focal=0, h4_quality=0.6, h6_focal=1)[arm]
                row = dict(train_seed=seed, eval_seed=e, arm=arm, steps=600, images=750, full_dataset=True,
                           measured_images=200, AP=66 + seed / 10 + e / 100 + gain, AP75=73 + gain,
                           latency_p50_ms=34 if arm == "h6_focal" else 28, latency_p95_ms=36,
                           **{"AP-common": 80 + gain, "AP-rare": 45 + gain})
                rows.append(row)
                if arm == "h4_focal":
                    reference.append(dict(row, arm="uniform"))
    return dict(rows=rows, manifest=manifest, completed=dict(evaluations=27, new_training_runs=6, reused_controls=3),
                reference=reference)


def test_quality_and_depth_effects_are_separate_and_gate_passes_valid_gain():
    data = evidence()
    metrics, discrepancy = audit_matrix(**data)
    pairs, categories, gate, depth = summarize(data["rows"], metrics)
    assert discrepancy == 0
    assert len(pairs) == 3 and len(categories) == 2
    assert gate["passed_development_screen"]
    assert gate["mean_delta_AP"] == pytest.approx(0.6)
    assert depth["mean_H4_minus_H6_AP"] == pytest.approx(-1)


@pytest.mark.parametrize("corruption", ["duplicate", "partial", "baseline", "nan", "seeds", "missing_category"])
def test_report_rejects_invalid_evidence(corruption):
    data = evidence()
    if corruption == "duplicate":
        data["rows"][-1] = data["rows"][0]
    elif corruption == "partial":
        data["rows"][0]["full_dataset"] = False
    elif corruption == "baseline":
        data["rows"][0]["AP"] += 0.01
    elif corruption == "nan":
        data["reference"][0]["AP"] = float("nan")
    elif corruption == "seeds":
        data["manifest"]["args"]["train_seeds"] = [10, 11]
    else:
        for row in data["rows"]:
            row.pop("AP-rare")
    with pytest.raises(ValueError):
        audit_matrix(**data)


@pytest.mark.parametrize("veto", ["AP", "AP75", "positive_pairs", "latency", "worst_category"])
def test_every_gate_can_reject_quality_candidate(veto):
    data = evidence()
    for row in data["rows"]:
        if row["arm"] != "h4_quality":
            continue
        if veto == "AP":
            row["AP"] -= 0.2
        elif veto == "AP75":
            row["AP75"] = 73
        elif veto == "positive_pairs":
            row["AP"] += 3 if row["train_seed"] == 10 else -0.7
        elif veto == "latency":
            row["latency_p50_ms"] = 30
        else:
            row["AP-rare"] = 42.9
    metrics, _ = audit_matrix(**data)
    _, _, gate, _ = summarize(data["rows"], metrics)
    assert not gate["checks"][veto]
    assert not gate["passed_development_screen"]
    assert sum(gate["checks"].values()) == 4


def training_evidence():
    logs = []
    for step in range(600):
        record = dict(input_digest="input", source_digest="noise", base_timesteps=[12]*8,
                      gates=[0.3]*8, rng_after_forward_digest="rng")
        digest = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
        logs.append(dict(step=step, loss_total=1.0, lr=2.5e-5*lr_factor(step), timesteps=[12]*8,
                         pair_digest=digest, **record, quality_positives=8, quality_mean_iou=0.8))
    info = dict(executed_steps=600, initial_state_sha256="init", initial_h4_prefix_sha256="init",
                final_class_loss="quality_focal", **audit_log(logs, "uniform", 0.5, 8))
    reference = [{k: v for k, v in r.items() if not k.startswith("quality_")} for r in logs]
    return logs, reference, info


def test_training_audit_checks_all_600_steps_and_allows_an_empty_batch():
    logs, reference, info = training_evidence()
    logs[5]["quality_positives"] = 0
    logs[5]["quality_mean_iou"] = 0
    result = audit_training(logs, reference, info, "init")
    assert result["image_passes"] == 4800


@pytest.mark.parametrize("field,value", [("lr", 0.1), ("source_digest", "other"),
                                          ("quality_mean_iou", float("nan"))])
def test_training_audit_rejects_unpaired_or_invalid_logs(field, value):
    logs, reference, info = training_evidence()
    logs[-1][field] = value
    with pytest.raises(ValueError):
        audit_training(logs, reference, info, "init")


def test_config_comparison_only_allows_known_disabled_fields_and_warmup_metadata():
    old = dict(OUTPUT_DIR="old", SOLVER=dict(WARMUP_FACTOR=0.01, BASE_LR=2.5e-5),
               MODEL=dict(DiffusionDet=dict(NUM_HEADS=4)))
    new = copy.deepcopy(old)
    new["OUTPUT_DIR"] = "new"
    new["SOLVER"]["WARMUP_FACTOR"] = 0.1
    new["MODEL"]["DiffusionDet"].update(FINAL_CLASS_LOSS="focal", INFERENCE_TOPK=0)
    check_config(new, old)
    new["SOLVER"]["BASE_LR"] = 1e-4
    with pytest.raises(ValueError):
        check_config(new, old)


def test_smoke_learning_rate_is_prefix_of_full_600_step_schedule():
    assert lr_factor(0) == 0.1
    assert lr_factor(20) == 1
    assert lr_factor(479) == 1
    assert lr_factor(480) == 0.1
