# Copyright (c) Facebook, Inc. and its affiliates.
# Adapted from detectron2.modeling.poolers.ROIPooler; installed code is unchanged.
"""Opt-in eager ROI pooling with batch indices built from known Python counts."""
import torch

from detectron2.layers import nonzero_tuple
from detectron2.modeling.poolers import ROIPooler, _create_zeros, assign_boxes_to_levels
from detectron2.utils.tracing import is_fx_tracing


def known_count_pooler_format(box_lists):
    """Avoid copying repeat counts to CUDA and reading dynamic output lengths.

    Shape metadata supplies the counts; no box coordinates or level assignments
    move to CPU. This also handles different counts and empty images in a batch.
    """
    boxes = torch.cat([box.tensor for box in box_lists], dim=0)
    indices = torch.cat([box.tensor.new_full((len(box), 1), index)
                         for index, box in enumerate(box_lists)], dim=0)
    return torch.cat([indices, boxes], dim=1)


class KnownCountROIPooler(ROIPooler):
    def forward(self, x, box_lists):
        # Python shape counts are eager-only. Export retains upstream behavior.
        if torch.jit.is_tracing() or is_fx_tracing():
            return super().forward(x, box_lists)
        torch._assert(isinstance(x, list) and isinstance(box_lists, list),
                      'Arguments to pooler must be lists')
        torch._assert(len(x) == len(self.level_poolers), 'Feature pyramid level count differs')
        torch._assert(len(box_lists) == x[0].size(0), 'Feature and box batch sizes differ')
        if len(box_lists) == 0:
            return _create_zeros(None, x[0].shape[1], *self.output_size, x[0])
        formatted = known_count_pooler_format(box_lists)
        if len(self.level_poolers) == 1:
            return self.level_poolers[0](x[0], formatted)
        assignments = assign_boxes_to_levels(box_lists, self.min_level, self.max_level,
                                             self.canonical_box_size, self.canonical_level)
        # Keep upstream allocation, level selection and index_put_ unchanged.
        size = self.output_size[0]
        output = _create_zeros(formatted, x[0].shape[1], size, size, x[0])
        for level, pooler in enumerate(self.level_poolers):
            indices = nonzero_tuple(assignments == level)[0]
            output.index_put_((indices,), pooler(x[level], formatted[indices]))
        return output
