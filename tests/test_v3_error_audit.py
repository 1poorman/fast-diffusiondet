import numpy as np

from scripts.v3_error_audit import audit_image, box_iou, spearman


def test_rank_correlation_handles_ties_and_constant_vectors():
    np.testing.assert_allclose(spearman([1,2,2,4],[4,3,3,1]), -1)
    assert np.isnan(spearman([1,1],[2,3]))


def test_iou_geometry_and_empty_sets():
    np.testing.assert_allclose(box_iou([[0,0,10,10]],[[0,0,10,10],[5,0,10,10],[20,20,1,1]]),[[1,1/3,0]])
    assert box_iou([],[[0,0,1,1]]).shape == (0,1)
    assert box_iou([[0,0,1,1]],[]).shape == (1,0)


def test_matching_duplicates_localization_confusion_and_empty_images():
    gt=[dict(bbox=[0,0,10,10],category_id=1)]
    boxes=[[0,0,10,10],[0,0,10,10],[0,0,10,10],[5,0,10,10],[30,30,10,10]]
    preds=[dict(bbox=b,category_id=2 if j==2 else 1,score=1-0.1*j) for j,b in enumerate(boxes)]
    counts,_,_=audit_image(preds,gt,0.3)
    assert all(counts[k]==1 for k in ("matched","duplicate","classification","localization","background","covered_gt75"))
    counts,_,_=audit_image([],gt,0.3)
    assert counts["covered_gt50"] == 0 and counts["gt"] == 1
    counts,_,_=audit_image(preds,[],0.3)
    assert counts["background"] == counts["predictions"] == 5
