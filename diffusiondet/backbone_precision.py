"""Opt-in inference autocast for the backbone, with an FP32 feature boundary."""
import torch


def backbone_features(backbone, images, precision='fp32', *, training=False):
    if precision not in ('fp32', 'fp16'):
        raise ValueError('EVAL_BACKBONE_PRECISION must be fp32 or fp16')
    # Keep the historical training/ambient-autocast behavior, even if inference
    # precision was selected in a config used to resume training.
    if training or precision == 'fp32':
        return backbone(images)
    if images.device.type != 'cuda' or images.dtype != torch.float32:
        raise ValueError('Backbone-only FP16 requires CUDA FP32 input images')
    if torch.is_autocast_enabled():
        raise ValueError('Backbone-only FP16 requires outer CUDA autocast disabled')
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        features = backbone(images)
    # ROIAlign, attention, regression and postprocessing keep the original dtype.
    # Parameters and buffers remain FP32; checkpoint keys and training are intact.
    return {name: feature.float() for name, feature in features.items()}
