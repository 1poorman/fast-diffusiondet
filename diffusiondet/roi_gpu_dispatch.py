"""Optional CUDA inference dispatch; autograd and unsupported cases stay native."""
import torch
from detectron2.modeling.poolers import ROIPooler, _create_zeros, assign_boxes_to_levels
from detectron2.utils.tracing import is_fx_tracing

from .roi_pooler import known_count_pooler_format


class GPUDispatchROIPooler(ROIPooler):
    def __init__(self, output_size, scales, sampling_ratio, pooler_type,
                 canonical_box_size=224, canonical_level=4):
        super().__init__(output_size, scales, sampling_ratio, pooler_type, canonical_box_size, canonical_level)
        self.dispatch_scales = tuple(scales)
        self.dispatch_supported = (pooler_type == 'ROIAlignV2' and sampling_ratio == 2
                                   and len(scales) == 4 and self.output_size[0] == self.output_size[1])

    def forward(self, x, box_lists):
        if (self.training or torch.is_grad_enabled() or torch.jit.is_tracing() or is_fx_tracing()
                or not self.dispatch_supported or torch.is_autocast_enabled()):
            return super().forward(x, box_lists)
        torch._assert(isinstance(x, list) and isinstance(box_lists, list), 'Arguments to pooler must be lists')
        torch._assert(len(x) == len(self.level_poolers), 'Feature pyramid level count differs')
        torch._assert(len(box_lists) == x[0].size(0), 'Feature and box batch sizes differ')
        if any(t.device.type != 'cuda' or t.device != x[0].device or t.dtype != torch.float32
               or not t.is_contiguous() or t.shape[:2] != x[0].shape[:2]
               or min(t.shape[-2:]) == 0 for t in x):
            return super().forward(x, box_lists)
        if not box_lists:
            return _create_zeros(None, x[0].shape[1], *self.output_size, x[0])
        if any(box.tensor.device != x[0].device or box.tensor.dtype != torch.float32 for box in box_lists):
            return super().forward(x, box_lists)
        from .roi_gpu_dispatch_kernel import dispatch_roi_align
        formatted = known_count_pooler_format(box_lists)
        levels = assign_boxes_to_levels(box_lists, self.min_level, self.max_level,
                                        self.canonical_box_size, self.canonical_level)
        return dispatch_roi_align(x, formatted, levels, self.output_size, self.dispatch_scales)
