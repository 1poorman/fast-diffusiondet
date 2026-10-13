from types import SimpleNamespace

import pytest
import torch

from diffusiondet.detector import DiffusionDet
from diffusiondet.proposal_budget import proposal_noise


def test_noise_prefix_and_rng_state_match_all_budgets_and_legacy_pool():
    torch.manual_seed(71)
    legacy = torch.randn(2, 300, 4)
    legacy_rng = torch.get_rng_state()
    for count in (300,200,150,100):
        torch.manual_seed(71)
        actual = proposal_noise(2, count, 300, "cpu")
        assert torch.equal(actual, legacy[:, :count])
        assert torch.equal(torch.get_rng_state(), legacy_rng)
        assert actual.is_contiguous()


@pytest.mark.parametrize("count", [0, -1, 301])
def test_noise_rejects_invalid_budget(count):
    with pytest.raises(ValueError):
        proposal_noise(1,count,300,"cpu")


@pytest.mark.parametrize("topk,expected", [(0,3),(2,2),(100,6)])
def test_output_topk_can_be_fixed_and_is_capped_by_available_scores(topk, expected):
    model = SimpleNamespace(device=torch.device("cpu"),num_classes=2, num_proposals=3, inference_topk=topk,
                            use_focal=True, use_fed_loss=False, use_nms=False, use_ensemble=False)
    logits = torch.tensor([[[4.,-3.],[2.,1.],[-2.,3.]]])
    boxes = torch.tensor([[[1.,2.,3.,4.],[5.,6.,7.,8.],[9.,10.,11.,12.]]])
    result = DiffusionDet.inference(model,logits,boxes,[(20,20)])[0]
    score, index = logits.sigmoid().flatten().topk(expected,sorted=False)
    assert torch.equal(result.scores,score)
    assert torch.equal(result.pred_classes,index % 2)
    assert torch.equal(result.pred_boxes.tensor,boxes[0,index // 2])


def test_injected_inference_noise_cannot_change_training_inputs():
    model = SimpleNamespace(training=True)
    with pytest.raises(ValueError,match="evaluation-only"):
        DiffusionDet.forward(model,[],inference_noise=torch.randn(1,3,4))


def test_coverage_rescales_clips_and_counts_crowded_and_category_groups():
    from collections import defaultdict
    import numpy as np
    from detectron2.structures import BoxMode
    from scripts.v3_proposal_sweep import accumulate_coverage
    record=dict(width=200,height=100,annotations=[
        dict(bbox=[20*i,0,10,10],bbox_mode=BoxMode.XYWH_ABS,category_id=i%2) for i in range(5)])
    boxes=np.array([[[10*i,0,10*i+5,5] for i in range(5)]]*4,dtype=float)
    acc=defaultdict(lambda:[0,0,0])
    accumulate_coverage(acc,boxes,record,(50,100))
    assert acc[(4,"all")] == [5,5,5]
    assert acc[(4,"crowded_ge5")] == [5,5,5]
    assert acc[(4,"category_0")] == [3,3,3]
    assert acc[(4,"category_1")] == [2,2,2]
    # Input diagnostics must not mutate the detector's returned coordinates.
    assert boxes[0,1,2] == 15
