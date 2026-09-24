# M1 Gate（一）：DDIM 等价 / NFE / 真实模型有限性
# conda activate fastdiff && python -m pytest tests/test_m1_ddim.py -v -x
import pytest
import torch

from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.structures import Instances

from diffusiondet import add_diffusiondet_config
from diffusiondet.solvers import run_sampler


def _build_model(**overrides):
    cfg = get_cfg()
    add_diffusiondet_config(cfg)
    cfg.merge_from_file("configs/diffdet.coco.res50.yaml")
    cfg.MODEL.DEVICE = "cpu"
    cfg.MODEL.WEIGHTS = ""
    for k, v in overrides.items():
        setattr(cfg.MODEL.DiffusionDet, k, v)
    model = build_model(cfg)
    model.eval()
    return model


def _prep(model, batched_inputs):
    images, images_whwh = model.preprocess_image(batched_inputs)
    src = model.backbone(images.tensor)
    features = [src[f] for f in model.in_features]
    return features, images_whwh, images


@pytest.mark.parametrize("sample_step", [1, 4])
def test_ddim_bit_exact(sample_step):
    """Gate 1: DDIM bit-exact (SAMPLE_STEP=1/4 covers renewal + ensemble)."""
    model = _build_model(SAMPLE_STEP=sample_step)
    inputs = [{"image": torch.rand(3, 256, 320)}]
    torch.manual_seed(42)
    features, whwh, images = _prep(model, inputs)
    with torch.no_grad():
        ref = model.ddim_sample(inputs, features, whwh, images)[0]["instances"]
    torch.manual_seed(42)
    with torch.no_grad():
        new = model(inputs)[0]["instances"]
    assert isinstance(new, Instances)
    diff = (new.pred_boxes.tensor - ref.pred_boxes.tensor).abs().max().item()
    assert diff < 1e-5, f"SAMPLE_STEP={sample_step}: box max|d|={diff}"
    assert torch.equal(new.scores, ref.scores)
    assert torch.equal(new.pred_classes, ref.pred_classes)


def test_nfe_ddim_real_model():
    """Gate 3: DDIM NFE == k on real model."""
    model = _build_model(SAMPLE_STEP=4)
    inputs = [{"image": torch.rand(3, 128, 160)}]
    features, whwh, images = _prep(model, inputs)
    holder = {}
    orig = model.make_denoise_fn

    def patched(feats, whwh_):
        fn = orig(feats, whwh_)
        holder["fn"] = fn
        return fn

    model.make_denoise_fn = patched
    torch.manual_seed(1)
    with torch.no_grad():
        run_sampler(model, inputs, features, whwh, images, do_postprocess=False)
    assert holder["fn"].nfe == 4, f"DDIM NFE={holder['fn'].nfe}, expect 4"


@pytest.mark.parametrize("solver", ["ddim", "euler", "heun", "dpm_v3"])
def test_real_model_finite(solver):
    """Gate 4: all solvers finite output, correct shape on real model."""
    model = _build_model(SAMPLE_STEP=4, BOX_RENEWAL=False, SOLVER=solver)
    inputs = [{"image": torch.rand(3, 128, 160)}]
    features, whwh, images = _prep(model, inputs)
    torch.manual_seed(7)
    with torch.no_grad():
        if solver == "ddim":
            # DDIM 走完整后处理（其逐 bit 路径保持原返回形式）
            results = run_sampler(model, inputs, features, whwh, images, do_postprocess=True)
            boxes = results[0]["instances"].pred_boxes.tensor
            assert boxes.shape[1] == 4 and torch.isfinite(boxes).all(), f"{solver}: boxes NaN/Inf"
        else:
            img, outputs_class, outputs_coord = run_sampler(
                model, inputs, features, whwh, images, do_postprocess=False)
            assert img.shape == (1, model.num_proposals, 4), img.shape
            assert torch.isfinite(img).all(), f"{solver}: img has NaN/Inf"
            assert torch.isfinite(outputs_coord[-1]).all(), f"{solver}: coords has NaN/Inf"
