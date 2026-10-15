"""Inference-only four-level ROIAlignV2 dispatch using the installed Triton runtime.

Bilinear interpolation follows torchvision ROIAlign's CUDA boundary convention.
The ROI's level is uniform within a program, so only its selected feature map is
sampled. No CPU-visible variable-length tensor is constructed.
"""
import torch
import triton
import triton.language as tl
from triton.language import math as tl_math


@triton.jit
def _interpolate(F, batch, channel, y, x, height, width, CHANNELS: tl.constexpr, mask):
    inside = (y >= -1.0) & (y <= height) & (x >= -1.0) & (x <= width) & mask
    y = tl.maximum(y, 0.0)
    x = tl.maximum(x, 0.0)
    yl, xl = y.to(tl.int32), x.to(tl.int32)
    edge_y, edge_x = yl >= height - 1, xl >= width - 1
    yl = tl.where(edge_y, height - 1, yl)
    xl = tl.where(edge_x, width - 1, xl)
    yh = tl.where(edge_y, yl, yl + 1)
    xh = tl.where(edge_x, xl, xl + 1)
    y = tl.where(edge_y, yl.to(tl.float32), y)
    x = tl.where(edge_x, xl.to(tl.float32), x)
    ly, lx = y - yl, x - xl
    hy, hx = 1.0 - ly, 1.0 - lx
    base = (batch * CHANNELS + channel) * height * width
    a = tl.load(F + base + yl * width + xl, inside, other=0.0)
    b = tl.load(F + base + yl * width + xh, inside, other=0.0)
    c = tl.load(F + base + yh * width + xl, inside, other=0.0)
    d = tl.load(F + base + yh * width + xh, inside, other=0.0)
    value = (hy * hx) * a + (hy * lx) * b + (ly * hx) * c + (ly * lx) * d
    return tl.where(inside, value, 0.0)


@triton.jit
def _sample(F, Boxes, Out, roi, height, width, CHANNELS: tl.constexpr,
            PH: tl.constexpr, PW: tl.constexpr, SCALE: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = index < CHANNELS * PH * PW
    pw = index % PW
    ph = (index // PW) % PH
    channel = index // (PH * PW)
    batch = tl.load(Boxes + roi * 5).to(tl.int32)
    start_x = tl.load(Boxes + roi * 5 + 1) * SCALE - 0.5
    start_y = tl.load(Boxes + roi * 5 + 2) * SCALE - 0.5
    end_x = tl.load(Boxes + roi * 5 + 3) * SCALE - 0.5
    end_y = tl.load(Boxes + roi * 5 + 4) * SCALE - 0.5
    bin_x = tl_math.div_rn(end_x - start_x, float(PW))
    bin_y = tl_math.div_rn(end_y - start_y, float(PH))
    value = tl.full((BLOCK,), 0.0, tl.float32)
    for iy in tl.static_range(2):
        y = start_y + ph * bin_y + (iy + 0.5) * bin_y / 2.0
        for ix in tl.static_range(2):
            x = start_x + pw * bin_x + (ix + 0.5) * bin_x / 2.0
            value += _interpolate(F, batch, channel, y, x, height, width, CHANNELS, mask)
    tl.store(Out + roi * CHANNELS * PH * PW + index, value / 4.0, mask)


@triton.jit(do_not_specialize=list(range(7, 15)))
def _dispatch(F0, F1, F2, F3, Boxes, Levels, Out,
              H0, W0, H1, W1, H2, W2, H3, W3,
              CHANNELS: tl.constexpr, PH: tl.constexpr, PW: tl.constexpr,
              S0: tl.constexpr, S1: tl.constexpr, S2: tl.constexpr, S3: tl.constexpr, BLOCK: tl.constexpr):
    roi = tl.program_id(0)
    level = tl.load(Levels + roi)
    if level == 0:
        _sample(F0, Boxes, Out, roi, H0, W0, CHANNELS, PH, PW, S0, BLOCK)
    elif level == 1:
        _sample(F1, Boxes, Out, roi, H1, W1, CHANNELS, PH, PW, S1, BLOCK)
    elif level == 2:
        _sample(F2, Boxes, Out, roi, H2, W2, CHANNELS, PH, PW, S2, BLOCK)
    elif level == 3:
        _sample(F3, Boxes, Out, roi, H3, W3, CHANNELS, PH, PW, S3, BLOCK)


def dispatch_roi_align(features, boxes, levels, output_size, scales):
    count, channels = boxes.shape[0], features[0].shape[1]
    # Match native zero output for invalid/unassigned levels and empty inputs.
    output = features[0].new_zeros((count, channels, *output_size))
    if count:
        dimensions = [value for feature in features for value in feature.shape[-2:]]
        _dispatch[(count, triton.cdiv(channels * output_size[0] * output_size[1], 256))](
            *features, boxes, levels, output, *dimensions, channels, *output_size, *scales,
            BLOCK=256, num_warps=4)
    return output
