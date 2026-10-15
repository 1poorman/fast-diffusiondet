"""GPU output/gradient audit and post-benchmark synchronization profiling for R1."""
from collections import Counter
from contextlib import contextmanager
import gzip
import json


def gradient_audit(device):
    import torch
    from detectron2.modeling.poolers import ROIPooler, convert_boxes_to_pooler_format
    from detectron2.structures import Boxes
    from diffusiondet.roi_pooler import KnownCountROIPooler, known_count_pooler_format
    torch.manual_seed(83)
    sizes = torch.tensor([0., 55.9, 56., 111.9, 112., 223.9, 224., 448.], device=device)
    coords = torch.stack((sizes * 0 - 8, sizes * 0 - 5, sizes - 8, sizes - 5), dim=1)
    boxes = [Boxes(coords), Boxes(coords[:0]), Boxes(coords[:3])]
    torch.testing.assert_close(convert_boxes_to_pooler_format(boxes), known_count_pooler_format(boxes), rtol=0, atol=0)
    cfg = dict(output_size=3, scales=[0.25, 0.125, 0.0625, 0.03125], sampling_ratio=2, pooler_type='ROIAlignV2')
    a = [torch.rand(3, 8, 64 // 2**i, 80 // 2**i, device=device, requires_grad=True) for i in range(4)]
    b = [x.detach().clone().requires_grad_(True) for x in a]
    expected, actual = ROIPooler(**cfg)(a, boxes), KnownCountROIPooler(**cfg)(b, boxes)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    expected.square().sum().backward()
    actual.square().sum().backward()
    differences = []
    for left, right in zip(a, b):
        if not torch.isfinite(right.grad).all():
            raise RuntimeError('Non-finite ROI feature gradient')
        torch.testing.assert_close(left.grad, right.grad, rtol=2e-5, atol=2e-5)
        differences.append(float((left.grad - right.grad).abs().max()))
    return dict(device=str(device), format_bit_exact=True, output_bit_exact=True,
                gradient_max_abs=max(differences), gradient_rtol=2e-5, gradient_atol=2e-5, passed=True)


def roi_events(events):
    scopes = [e for e in events if e.get('ph') == 'X' and e.get('name') == 'R1/roi_pool']
    scoped = [e for e in events if e.get('ph') == 'X' and any(
        s.get('pid') == e.get('pid') and s.get('tid') == e.get('tid')
        and s['ts'] <= e['ts'] and e['ts'] + e.get('dur', 0) <= s['ts'] + s['dur'] for s in scopes)]
    sync = Counter(e['name'] for e in scoped if e.get('cat') == 'cuda_runtime' and 'Synchronize' in e['name'])
    repeats = sum(e.get('cat') == 'cpu_op' and e['name'] == 'aten::repeat_interleave' for e in scoped)
    return dict(roi_calls=len(scopes), synchronizations=dict(sync), synchronization_count=sum(sync.values()),
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
        with torch.profiler.record_function('R1/' + name):
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
