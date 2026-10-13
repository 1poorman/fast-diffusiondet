"""Guard P1 decisions against partial runs, invalid pairing, and recall regressions."""
import copy

import pytest

from scripts.v3_summarize_proposals import audit_annotations, audit_matrix, summarize


def test_raw_and_evaluator_annotation_audit_allows_only_background_filtering():
    original = dict(images=[dict(id=1)], categories=[dict(id=0, name="_background_"), dict(id=1, name="plant")],
                    annotations=[dict(image_id=1, category_id=1, bbox=[1, 2, 3, 4])])
    evaluation = dict(info={}, licenses=[], images=original["images"], categories=[original["categories"][1]],
                      annotations=copy.deepcopy(original["annotations"]))
    audit_annotations(original, evaluation)
    evaluation["annotations"][0]["bbox"][0] = 2
    with pytest.raises(ValueError, match="annotations differ"):
        audit_annotations(original, evaluation)


def evidence():
    manifest = dict(args=dict(train_seeds=[10, 11, 12], eval_seeds=[0, 1, 2],
                              proposals=[300, 200, 150, 100], limit=0),
                    image_ids=list(range(750)), full_images=750, category_names=["common", "rare"])
    rows, coverage, reference = [], [], []
    for s in (10, 11, 12):
        for e in (0, 1, 2):
            for p in (300, 200, 150, 100):
                row = dict(train_seed=s, eval_seed=e, proposals=p, images=750, full_dataset=True,
                           measured_images=200, AP=65 + s / 10 + e / 100, AP75=73.0,
                           latency_p50_ms=30.0 if p == 300 else 24.0, latency_p95_ms=32.0,
                           **{"AP-common": 80.0, "AP-rare": 40.0})
                rows.append(row)
                if p == 300:
                    reference.append(dict(row, arm="uniform"))
                for stage in range(1, 5):
                    for group, n in (("all", 100), ("crowded_ge5", 50), ("category_0", 80), ("category_1", 20)):
                        coverage.append(dict(train_seed=s, eval_seed=e, proposals=p, stage=stage,
                                             group=group, objects=n, hits50=n, hits75=n,
                                             coverage50=1.0, coverage75=1.0))
    parity = [dict(train_seed=s, image_id=i, bit_exact=True) for s in (10, 11, 12) for i in (0, 1)]
    return dict(rows=rows, coverage=coverage, manifest=manifest, completed=dict(evaluations=36, checkpoints=3),
                parity=parity, reference=reference)


def test_complete_screen_averages_inference_within_checkpoint():
    data = evidence()
    audit, metrics = audit_matrix(**data)
    paired, summary, classes, gates = summarize(data["rows"], data["coverage"], data["manifest"], metrics)
    assert audit["evaluations"] == 36
    assert audit["baseline_max_abs_AP_difference"] == 0
    assert len(paired) == 12 and len(classes) == 8
    assert summary[1]["AP"] == pytest.approx(66.11)
    assert summary[1]["p50_reduction_percent"] == pytest.approx(20)
    assert all(g["passed_development_screen"] for g in gates)


@pytest.mark.parametrize("corruption", ["duplicate_eval", "subset", "missing_coverage", "duplicate_coverage",
                                        "bad_fraction", "changed_denominator", "nan", "reference_AP", "reference_nan", "parity"])
def test_report_rejects_corrupt_or_incomplete_evidence(corruption):
    data = copy.deepcopy(evidence())
    if corruption == "duplicate_eval":
        data["rows"][-1] = data["rows"][0]
    elif corruption == "subset":
        data["rows"][0]["full_dataset"] = False
    elif corruption == "missing_coverage":
        data["coverage"].pop()
    elif corruption == "duplicate_coverage":
        data["coverage"][-1] = data["coverage"][0]
    elif corruption == "bad_fraction":
        data["coverage"][0]["coverage50"] = 0.9
    elif corruption == "changed_denominator":
        data["coverage"][0].update(objects=50, hits50=50, hits75=50)
    elif corruption == "nan":
        data["rows"][0]["AP"] = float("nan")
    elif corruption == "reference_AP":
        data["reference"][0]["AP"] += 0.01
    elif corruption == "reference_nan":
        data["reference"][0]["AP"] = float("nan")
    else:
        data["parity"][0]["bit_exact"] = False
    with pytest.raises(ValueError):
        audit_matrix(**data)


@pytest.mark.parametrize("veto", ["AP", "AP75", "p50", "worst_category", "crowded_coverage50"])
def test_each_predeclared_gate_can_veto_an_otherwise_fast_candidate(veto):
    data = evidence()
    for row in data["rows"]:
        if row["proposals"] != 100:
            continue
        if veto in ("AP", "AP75"):
            row[veto] -= 0.31
        elif veto == "p50":
            row["latency_p50_ms"] = 27.1
        elif veto == "worst_category":
            row["AP-rare"] -= 2.01
    if veto == "crowded_coverage50":
        for row in data["coverage"]:
            if row["proposals"] == 100 and row["stage"] == 4 and row["group"] == "crowded_ge5":
                row.update(hits50=49, hits75=49, coverage50=0.98, coverage75=0.98)
    _, metrics = audit_matrix(**data)
    *_, gates = summarize(data["rows"], data["coverage"], data["manifest"], metrics)
    candidate = next(g for g in gates if g["proposals"] == 100)
    assert candidate["checks"][veto] is False
    assert not candidate["passed_development_screen"]
    assert sum(candidate["checks"].values()) == 4
