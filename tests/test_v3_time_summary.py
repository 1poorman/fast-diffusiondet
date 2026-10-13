"""The report must reject audit corruption before drawing a model conclusion."""
import copy
import hashlib
import json

import pytest

from scripts.v3_summarize_time_alignment import audit_log


def records(arm):
    row = dict(step=0, loss_total=1.0, input_digest="image", source_digest="noise",
               base_timesteps=[42, 999], gates=[0.25, 0.75], rng_after_forward_digest="rng",
               timesteps=[999,999] if arm == "mix_tmax" else [42,999])
    pair = {k:row[k] for k in ("input_digest", "source_digest", "base_timesteps", "gates", "rng_after_forward_digest")}
    row["pair_digest"] = hashlib.sha256(json.dumps(pair, sort_keys=True).encode()).hexdigest()
    return [row]


def test_valid_pair_shares_sources_but_has_different_actual_t():
    u = audit_log(records("uniform"), "uniform", 0.5, 2)
    m = audit_log(records("mix_tmax"), "mix_tmax", 0.5, 2)
    assert u["pairing_trace_sha256"] == m["pairing_trace_sha256"]
    assert u["tmax_fraction"] == 0.5
    assert m["tmax_fraction"] == 1.0
    assert u["image_passes"] == m["image_passes"] == 2


@pytest.mark.parametrize("field,value", [("timesteps", [42,999]), ("source_digest", "changed"),
                                         ("step", 2), ("loss_total", float("nan")), ("gates", [0.25])])
def test_report_rejects_inconsistent_trace(field, value):
    logs = copy.deepcopy(records("mix_tmax"))
    logs[0][field] = value
    with pytest.raises(ValueError):
        audit_log(logs, "mix_tmax", 0.5, 2)
