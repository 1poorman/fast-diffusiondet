# -*- coding: utf-8 -*-
"""
COCO 迁移初始化（决策 D2）。

为什么不能直接用 ``MODEL.WEIGHTS``：
``torch.nn.Module.load_state_dict(..., strict=False)`` 只忽略 missing/unexpected key，
**形状不匹配仍会抛 RuntimeError**。`diffdet_coco_res50.pth` 是 80 类的完整权重，
其分类层权重为 ``[80, 256]``，与本项目 16/7 类不一致 -> 直接加载必崩。

因此这里在加载前做逐参数形状过滤：形状一致才载入，不一致的记录下来并跳过
（那些层保留构造随机初始化），保证**所有 arm 使用同一份权重与同一套随机补充层**。

权重路径解析顺序（均不入库，见 .gitignore）：
    1. ``cfg.MODEL.DiffusionDet.PRETRAIN_PATH``（命令行 --opts 覆盖）
    2. 环境变量 ``FASTDD_PRETRAIN``
    3. ``<repo>/petrain/diffdet_coco_res50.pth``
    4. ``<repo>/../DiffusionDet-main/petrain/diffdet_coco_res50.pth``
"""

import os

import torch

__all__ = ["resolve_pretrain", "load_transfer_weights"]

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_FILENAME = "diffdet_coco_res50.pth"

_PATH_CANDIDATES = (
    os.path.join(_REPO_ROOT, "petrain", _DEFAULT_FILENAME),
    os.path.normpath(os.path.join(_REPO_ROOT, os.pardir, "DiffusionDet-main", "petrain", _DEFAULT_FILENAME)),
)


def resolve_pretrain(cfg=None):
    """返回可用的预训练权重路径，找不到返回 None。"""
    candidates = []
    if cfg is not None:
        p = getattr(cfg.MODEL.DiffusionDet, "PRETRAIN_PATH", "")
        if p:
            candidates.append(p)
    env = os.environ.get("FASTDD_PRETRAIN", "")
    if env:
        candidates.append(env)
    candidates.extend(_PATH_CANDIDATES)

    for c in candidates:
        if c and os.path.isfile(c):
            return os.path.normpath(c)
    return None


def _strip_module_prefix(state_dict):
    return {k[len("module."):] if k.startswith("module.") else k: v for k, v in state_dict.items()}


def load_transfer_weights(model, path=None, cfg=None, logger=None):
    """
    把 COCO checkpoint 中**形状匹配**的参数载入 model。

    Returns:
        dict: ``{"loaded": n, "skipped_shape": [...], "skipped_missing_in_model": n, "path": p}``
              若未提供不可用路径则返回 None（此时调用方应退回普通模型初始化）。
    """
    if path is None:
        path = resolve_pretrain(cfg)
    if path is None or not os.path.isfile(path):
        if logger is not None:
            logger.warning("[weights] 未找到 COCO 预训练权重，跳过迁移初始化（退回随机/ImageNet 初始化）")
        return None

    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt
    state_dict = _strip_module_prefix(state_dict)

    model_sd = model.state_dict()
    filtered, skipped_shape, missing_in_model = {}, [], 0
    for k, v in state_dict.items():
        if k not in model_sd:
            missing_in_model += 1
            continue
        if model_sd[k].shape != v.shape:
            skipped_shape.append((k, tuple(v.shape), tuple(model_sd[k].shape)))
            continue
        filtered[k] = v

    incompatible = model.load_state_dict(filtered, strict=False)

    info = {
        "path": path,
        "loaded": len(filtered),
        "skipped_shape": skipped_shape,
        "skipped_missing_in_model": missing_in_model,
        "unexpected_in_model": list(incompatible.unexpected_keys)[:20],
    }
    if logger is not None:
        logger.info("[weights] 迁移初始化 <- {}".format(path))
        logger.info("[weights]   载入 {} / {} 个参数".format(len(filtered), len(state_dict)))
        if skipped_shape:
            logger.info("[weights]   因形状不匹配而重新初始化 {} 个参数（多为分类层）:".format(len(skipped_shape)))
            for k, s_ckpt, s_model in skipped_shape[:10]:
                logger.info("[weights]     {:<70} ckpt{} != model{}".format(k, s_ckpt, s_model))
        if missing_in_model:
            logger.info("[weights]   checkpoint 中 {} 个参数在本模型中不存在（已忽略）".format(missing_in_model))
    return info


def maybe_freeze_backbone_stage1(model, cfg):
    """A11：冻结 stem + res2，缓解 6GB 显存压力。默认关闭。"""
    if cfg is not None and getattr(cfg.MODEL.DiffusionDet, "FREEZE_BACKBONE_STAGE1", False):
        frozen = 0
        for mod in (getattr(model.backbone, "bottom_up", None),):
            if mod is None:
                continue
        bb = getattr(model.backbone, "bottom_up", model.backbone)
        for attr in ("stem", "conv1", "bn1", "layer1", "res2"):
            m = getattr(bb, attr, None)
            if m is not None:
                for p in m.parameters():
                    p.requires_grad = False
                    frozen += 1
                break  # stem/layer1 二选一即可覆盖
        return frozen
    return 0
