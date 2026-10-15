"""R2 ROI scopes, dynamic-index synchronization and correlated kernel counts."""
from collections import Counter
from contextlib import contextmanager
import gzip
import json


def roi_events(events):
    scopes = [e for e in events if e.get('ph') == 'X' and e.get('name') == 'R2/roi_pool']
    scoped = [e for e in events if e.get('ph') == 'X' and any(
        s.get('pid') == e.get('pid') and s.get('tid') == e.get('tid')
        and s['ts'] <= e['ts'] and e['ts'] + e.get('dur', 0) <= s['ts'] + s['dur'] for s in scopes)]
    sync = Counter(e['name'] for e in scoped if e.get('cat') == 'cuda_runtime' and 'Synchronize' in e['name'])
    repeats = sum(e.get('cat') == 'cpu_op' and e['name'] == 'aten::repeat_interleave' for e in scoped)
    nonzero = sum(e.get('cat') == 'cpu_op' and e['name'] == 'aten::nonzero' for e in scoped)
    correlations = {e.get('args', {}).get('correlation') for e in scoped if e.get('cat') in ('cuda_runtime', 'cuda_driver')}
    correlations.discard(None)
    kernels = Counter(e['name'] for e in events if e.get('cat') == 'kernel' and e.get('args', {}).get('correlation') in correlations)
    return dict(nonzero_calls=nonzero, kernel_count=sum(kernels.values()), kernels=dict(kernels), roi_calls=len(scopes), synchronizations=dict(sync), synchronization_count=sum(sync.values()),
                repeat_interleave_calls=repeats)


def profile_model(model, loader, invoke, seed, reference, target):
    import torch
    from scripts.v3_profile_utils import ScopedCalls, prediction_digest
    batches = []
    for i, batch in enumerate(loader):
        if i >= 30:
            break
        if i >= 20:
            batches.append(batch)

    @contextmanager
    def scope(name):
        with torch.profiler.record_function('R2/' + name):
            yield

    # Warm both the model and allocator after reloading; profiling is after all timing.
    for batch in batches:
        invoke(model, batch, seed)
    actual = []
    with ScopedCalls(scope) as wrappers:
        wrappers.wrap(model.head.box_pooler, 'forward', 'roi_pool')
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
            for batch in batches:
                prediction = invoke(model, batch, seed)
                actual.append(prediction_digest(prediction))
    expected = [reference[str(b[0]['image_id'])] for b in batches]
    if actual != expected:
        raise RuntimeError('Profiler changed predictions')
    raw = target.with_suffix('.json')
    prof.export_chrome_trace(str(raw))
    trace = json.loads(raw.read_text())
    with gzip.open(target, 'wt') as stream:
        json.dump(trace, stream)
    # Retain the raw trace as well; it is an ignored run artifact.
    return dict(**roi_events(trace['traceEvents']), images=len(batches), prediction_bit_exact=True)
