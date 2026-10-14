"""Temporary nested timing scopes; CPU dispatch and CUDA stream spans are distinct."""
from collections import defaultdict
from contextlib import contextmanager
import functools
import hashlib
import math
import time


def exclusive_rows(spans):
    """Partition each nested interval by subtracting only its direct children."""
    children = defaultdict(list)
    for index, span in enumerate(spans):
        parent = span['parent']
        if parent is not None:
            if not 0 <= parent < index:
                raise ValueError('Invalid timing tree')
            children[parent].append(span)
    result = []
    for index, span in enumerate(spans):
        row = dict(span)
        for field in ('host_ms', 'stream_ms'):
            value = float(span[field])
            remaining = value - sum(child[field] for child in children[index])
            if not math.isfinite(value) or value < 0 or remaining < -0.02:
                raise ValueError('Invalid nested duration')
            row['self_' + field] = remaining
        result.append(row)
    return result


class ScopedCalls:
    """Restore instance attributes exactly, including inherited methods, on exit."""
    def __init__(self, scope):
        self.scope = scope
        self.originals = []

    def wrap(self, obj, attr, label):
        original = getattr(obj, attr)
        existed = attr in vars(obj)
        saved = vars(obj).get(attr)

        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            with self.scope(label):
                return original(*args, **kwargs)

        self.originals.append((obj, attr, existed, saved))
        setattr(obj, attr, wrapped)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        for obj, attr, existed, saved in reversed(self.originals):
            if existed:
                setattr(obj, attr, saved)
            else:
                delattr(obj, attr)


class EventScopes:
    def __init__(self):
        self.spans, self.stack = [], []

    @contextmanager
    def scope(self, label):
        import torch
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        index = len(self.spans)
        row = dict(name=label, parent=self.stack[-1] if self.stack else None)
        self.spans.append(row)
        self.stack.append(index)
        start.record()
        tick = time.perf_counter()
        try:
            yield
        finally:
            row['host_ms'] = 1000 * (time.perf_counter() - tick)
            end.record()
            row['events'] = (start, end)
            self.stack.pop()

    def finish(self):
        # Caller has synchronized once, after the entire model invocation.
        if self.stack:
            raise ValueError('Unclosed timing scopes')
        for row in self.spans:
            start, end = row.pop('events')
            row['stream_ms'] = start.elapsed_time(end)
        return exclusive_rows(self.spans)


def instrument(model, scopes):
    """All patches live inside ScopedCalls; no model parameter or arithmetic changes."""
    import importlib
    scopes.wrap(model, 'preprocess_image', 'preprocess')
    scopes.wrap(model.backbone, 'forward', 'backbone')
    scopes.wrap(model.head, 'forward', 'head')
    scopes.wrap(model.head.box_pooler, 'forward', 'roi_pool')
    scopes.wrap(model, 'inference', 'selection_nms')
    scopes.wrap(importlib.import_module('diffusiondet.solvers.ddim'), 'detector_postprocess', 'postprocess')
    for index, stage in enumerate(model.head.head_series, 1):
        scopes.wrap(stage, 'forward', f'stage{index}')
        scopes.wrap(stage.self_attn, 'forward', 'attention')
        scopes.wrap(stage.inst_interact, 'forward', 'dynamic_conv')
        scopes.wrap(stage, 'apply_deltas', 'box_decode')


def prediction_digest(predictions):
    digest = hashlib.sha256()
    for output in predictions:
        inst = output['instances']
        for tensor in (inst.pred_boxes.tensor, inst.scores, inst.pred_classes):
            array = tensor.detach().cpu().contiguous().numpy()
            digest.update(str((array.shape, array.dtype)).encode())
            digest.update(array.tobytes())
    return digest.hexdigest()


def interval_union(intervals):
    """Union duration, avoiding double-counting concurrent CUDA activity."""
    total, end = 0.0, -math.inf
    for start, stop in sorted(intervals):
        if not math.isfinite(start) or not math.isfinite(stop) or stop < start:
            raise ValueError('Invalid trace interval')
        total += max(0.0, stop - max(end, start))
        end = max(end, stop)
    return total


def sync_scope_rows(trace):
    """Assign CPU CUDA synchronization calls to the innermost named host scope."""
    scopes = [e for e in trace if e.get('ph') == 'X' and e.get('cat') == 'user_annotation'
              and e.get('name', '').startswith('L1/')]
    result = []
    for event in trace:
        if event.get('cat') != 'cuda_runtime' or event.get('ph') != 'X' or 'Synchronize' not in event.get('name', ''):
            continue
        parents = [p for p in scopes if (p['pid'], p['tid']) == (event['pid'], event['tid'])
                   and p['ts'] <= event['ts'] and event['ts'] + event['dur'] <= p['ts'] + p['dur'] + 0.01]
        parent = min(parents, key=lambda p: p['dur'])['name'] if parents else 'outside_call'
        result.append(dict(scope=parent, runtime=event['name'], host_us=event['dur']))
    return result
