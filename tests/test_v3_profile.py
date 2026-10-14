from contextlib import contextmanager
import copy

import pytest

from scripts.v3_profile_utils import ScopedCalls, exclusive_rows, interval_union, sync_scope_rows
from scripts.v3_summarize_latency import audit_samples, sync_operators


def test_nested_exclusive_spans_partition_root_without_double_counting():
    rows = exclusive_rows([
        dict(name='call', parent=None, host_ms=20., stream_ms=18.),
        dict(name='head', parent=0, host_ms=12., stream_ms=11.),
        dict(name='roi', parent=1, host_ms=7., stream_ms=6.),
        dict(name='backbone', parent=0, host_ms=5., stream_ms=5.),
    ])
    assert [r['self_host_ms'] for r in rows] == [3., 5., 7., 5.]
    assert sum(r['self_stream_ms'] for r in rows) == 18.


@pytest.mark.parametrize('row', [
    dict(name='bad', parent=0, host_ms=1, stream_ms=1),
    dict(name='bad', parent=None, host_ms=float('nan'), stream_ms=1),
])
def test_invalid_timing_evidence_is_rejected(row):
    with pytest.raises(ValueError):
        exclusive_rows([row])


def test_child_cannot_take_longer_than_parent():
    with pytest.raises(ValueError):
        exclusive_rows([dict(name='root', parent=None, host_ms=1, stream_ms=1),
                        dict(name='child', parent=0, host_ms=2, stream_ms=1)])


def test_temporary_wrappers_restore_attributes_after_exception():
    calls = []

    @contextmanager
    def scope(name):
        calls.append(name)
        yield

    class Example:
        def operation(self, value):
            return value * 2

    obj = Example()
    obj.existing = lambda: 7
    before = dict(vars(obj))
    with pytest.raises(RuntimeError):
        with ScopedCalls(scope) as wrappers:
            wrappers.wrap(obj, 'operation', 'method')
            wrappers.wrap(obj, 'existing', 'existing')
            assert obj.operation(3) == 6 and obj.existing() == 7
            raise RuntimeError('failure in profiled call')
    assert vars(obj) == before and calls == ['method', 'existing']


def test_cuda_activity_union_handles_overlap_and_empty_trace():
    assert interval_union([(0, 3), (2, 5), (6, 7), (1, 2)]) == 6
    assert interval_union([]) == 0
    with pytest.raises(ValueError):
        interval_union([(3, 1)])


def test_sync_attribution_uses_innermost_scope_on_same_host_thread():
    common = dict(ph='X', pid=1, tid=2)
    trace = [dict(common, cat='user_annotation', name='L1/call', ts=0, dur=20),
             dict(common, cat='user_annotation', name='L1/roi_pool', ts=3, dur=10),
             dict(common, cat='cuda_runtime', name='cudaStreamSynchronize', ts=5, dur=2),
             dict(common, cat='cuda_runtime', name='cudaDeviceSynchronize', ts=21, dur=1),
             dict(common, cat='cuda_runtime', name='cudaStreamSynchronize', ts=5, dur=2, tid=3)]
    assert [r['scope'] for r in sync_scope_rows(trace)] == ['L1/roi_pool', 'outside_call', 'outside_call']
    trace.append(dict(common, cat='cpu_op', name='aten::nonzero', ts=4, dur=4))
    mapped = sync_operators(trace)
    assert [(r['scope'], r['operator']) for r in mapped] == [
        ('L1/roi_pool', 'aten::nonzero'), ('outside_call', 'outside_operator'), ('outside_call', 'outside_operator')]


def sample_evidence():
    spans = []

    def span(name, parent, duration):
        spans.append(dict(name=name, parent=parent, host_ms=duration, stream_ms=duration))
        return len(spans) - 1

    span('call', None, 26)
    span('preprocess', 0, 1)
    span('backbone', 0, 1)
    head = span('head', 0, 21)
    for index in range(1, 5):
        stage = span(f'stage{index}', head, 5)
        for name in ('roi_pool', 'attention', 'dynamic_conv', 'box_decode'):
            span(name, stage, 1)
    span('selection_nms', 0, 1)
    span('postprocess', 0, 1)
    plain = [dict(image_id=i, index=20 + i, prediction_sha256='unchanged', input_hw=[800, 1024], wall_ms=27.) for i in (0, 1)]
    events = [dict(row, spans=copy.deepcopy(exclusive_rows(spans))) for row in plain]
    return plain, events


def test_realistic_profile_tree_and_paired_outputs_pass():
    plain, events = sample_evidence()
    audit_samples(plain, events, [0, 1], 4)


@pytest.mark.parametrize('corruption', ['output', 'window', 'counts', 'exclusive', 'nonfinite'])
def test_profile_audit_rejects_corrupted_evidence(corruption):
    plain, events = sample_evidence()
    if corruption == 'output':
        events[0]['prediction_sha256'] = 'changed'
    elif corruption == 'window':
        events.reverse()
    elif corruption == 'counts':
        events[0]['spans'][4]['name'] = 'stage6'
    elif corruption == 'exclusive':
        events[0]['spans'][0]['self_stream_ms'] += 0.2
    else:
        plain[0]['wall_ms'] = float('nan')
    with pytest.raises(ValueError):
        audit_samples(plain, events, [0, 1], 4)
