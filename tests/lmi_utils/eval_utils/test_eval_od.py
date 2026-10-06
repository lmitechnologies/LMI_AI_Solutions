import numpy as np
import pytest

from lmi_utils.dataset_utils.representations import (
    Box,
    BoxAnnotation,
    Dataset,
    FileAnnotations,
    Label,
    Mask,
    MaskAnnotation,
    Polygon,
    PolygonAnnotation,
)
from lmi_utils.eval_utils.eval_od import curves, evaluate, has_masks, iou_matrix, match, pair_files, to_coco_geometry


def box(i, label, x0, y0, x1, y1, conf=None):
    return BoxAnnotation(id=str(i), label_id=label, value=Box(x0, y0, x1, y1), confidence=conf)


def dataset(files):
    return Dataset(labels=[Label(id="a"), Label(id="b")], files=files)


def test_counts_per_class_and_all():
    gt = FileAnnotations(
        id="0",
        path="img.png",
        height=100,
        width=100,
        annotations=[box(0, "a", 0, 0, 10, 10), box(1, "a", 50, 50, 60, 60), box(2, "b", 20, 20, 30, 30)],
    )
    pred = FileAnnotations(
        id="0",
        path="sub/img.png",  # paired by file name
        height=100,
        width=100,
        predictions=[
            box(0, "a", 0, 0, 10, 11, conf=0.9),  # tp
            box(1, "a", 0, 0, 10, 10, conf=0.8),  # duplicate of a matched object: fp
            box(2, "b", 50, 50, 60, 60, conf=0.7),  # wrong class: fp
            box(3, "b", 20, 20, 30, 30, conf=0.1),  # below confidence: ignored, so its object is a fn
        ],
    )
    m = evaluate(dataset([gt]), dataset([pred]), iou_thres=0.5, conf_thres=0.5)

    assert (m["a"]["tp"], m["a"]["fp"], m["a"]["fn"]) == (1, 1, 1)
    assert (m["b"]["tp"], m["b"]["fp"], m["b"]["fn"]) == (0, 1, 1)
    assert (m["all"]["tp"], m["all"]["fp"], m["all"]["fn"]) == (1, 2, 2)
    assert m["all"]["precision"] == pytest.approx(1 / 3)
    assert m["all"]["recall"] == pytest.approx(1 / 3)
    assert m["all"]["f1"] == pytest.approx(1 / 3)
    assert m["b"]["f1"] == 0.0


def test_mask_iou_differs_from_box_iou():
    triangle = PolygonAnnotation(id="0", label_id="a", value=Polygon([[0, 0], [0, 39], [39, 39]]))  # fills half its box
    gt = FileAnnotations(id="0", path="img.png", height=100, width=100, annotations=[triangle])
    pred = FileAnnotations(id="0", path="img.png", height=100, width=100, predictions=[box(0, "a", 0, 0, 39, 39, conf=0.9)])

    by_box = evaluate(dataset([gt]), dataset([pred]), iou_thres=0.7)["a"]
    by_mask = evaluate(dataset([gt]), dataset([pred]), iou_thres=0.7, use_mask=True)["a"]
    assert (by_box["tp"], by_box["fp"], by_box["fn"]) == (1, 0, 0)
    assert (by_mask["tp"], by_mask["fp"], by_mask["fn"]) == (0, 1, 1)


def test_mask_prediction_matches_its_own_mask():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10:30, 40:70] = 1
    gt = FileAnnotations(id="0", path="img.png", height=100, width=100, annotations=[box(0, "a", 40, 10, 70, 30)])
    pred = FileAnnotations(
        id="0", path="img.png", height=100, width=100, predictions=[MaskAnnotation(id="0", label_id="a", value=Mask(mask), confidence=0.9)]
    )
    for use_mask in (False, True):
        assert evaluate(dataset([gt]), dataset([pred]), iou_thres=0.9, use_mask=use_mask)["all"]["f1"] == 1.0


@pytest.mark.parametrize("region", [(slice(10, 30), slice(40, 70)), (slice(5, 6), slice(7, 8)), (slice(99, 100), slice(0, 120))])
def test_mask_box_matches_mask_to_box(region):
    mask = np.zeros((100, 120), dtype=np.uint8)
    mask[region] = 1
    annotation = MaskAnnotation(id="0", label_id="a", value=Mask(mask))
    assert to_coco_geometry(annotation, 100, 120, use_mask=False) == annotation.value.to_box(h=100, w=120).to_coco()


def test_mask_rle_iou_matches_pixel_iou():
    a, b = np.zeros((50, 80), dtype=np.uint8), np.zeros((50, 80), dtype=np.uint8)
    a[5:30, 10:60], b[20:45, 40:75] = 1, 1
    pixel_iou = (a & b).sum() / (a | b).sum()
    geometry = [to_coco_geometry(MaskAnnotation(id="0", label_id="a", value=Mask(m)), 50, 80, use_mask=True) for m in (a, b)]
    assert iou_matrix(geometry[:1], geometry[1:], use_mask=True)[0, 0] == pytest.approx(pixel_iou)


def test_has_masks_needs_shapes_on_both_sides(caplog):
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:5, 2:5] = 1
    polygon = PolygonAnnotation(id="0", label_id="a", value=Polygon([[0, 0], [0, 5], [5, 5]]))
    mask_pred = MaskAnnotation(id="0", label_id="a", value=Mask(mask), confidence=0.9)

    def pairs(annotations, predictions):
        gt = FileAnnotations(id="0", path="img.png", height=10, width=10, annotations=annotations)
        pred = FileAnnotations(id="0", path="img.png", height=10, width=10, predictions=predictions)
        return pair_files(dataset([gt]), dataset([pred]))

    def check(annotations, predictions, expected, warns):
        caplog.clear()
        assert has_masks(pairs(annotations, predictions)) == expected
        assert ("boxes and masks are mixed" in caplog.text) == warns

    box_label, box_pred = box(1, "a", 0, 0, 5, 5), box(1, "a", 0, 0, 5, 5, conf=0.9)
    check([polygon], [mask_pred], expected=True, warns=False)
    polygon_pred = PolygonAnnotation(id="2", label_id="a", value=Polygon([[0, 0], [0, 5], [5, 5]]), confidence=0.9)
    check([polygon], [polygon_pred], expected=True, warns=False)
    check([polygon], [], expected=False, warns=False)
    check([box_label], [box_pred], expected=False, warns=False)
    check([polygon], [box_pred], expected=False, warns=True)
    check([box_label], [mask_pred], expected=False, warns=True)
    check([polygon, box_label], [mask_pred], expected=False, warns=True)
    check([polygon], [mask_pred, box_pred], expected=False, warns=True)


def test_curves_match_evaluating_at_each_confidence():
    gt = FileAnnotations(
        id="0", path="img.png", height=100, width=100, annotations=[box(0, "a", 0, 0, 10, 10), box(1, "a", 50, 50, 60, 60)]
    )
    pred = FileAnnotations(
        id="0",
        path="img.png",
        height=100,
        width=100,
        predictions=[
            box(0, "a", 0, 0, 10, 10, conf=0.9),
            box(1, "a", 20, 20, 30, 30, conf=0.6),
            box(2, "a", 50, 50, 60, 60, conf=0.3),
        ],
    )
    confidences = [0.0, 0.3, 0.5, 0.9, 1.0]
    c = curves(match(pair_files(dataset([gt]), dataset([pred]))), confidences)["all"]

    assert c["tp"].tolist() == [2, 2, 1, 1, 0]
    assert c["fp"].tolist() == [1, 1, 1, 0, 0]
    assert c["fn"].tolist() == [0, 0, 1, 1, 2]
    assert np.isnan(c["precision"][-1])  # no predictions kept
    for k, conf in enumerate(confidences):
        at = evaluate(dataset([gt]), dataset([pred]), conf_thres=conf)["all"]
        assert (at["tp"], at["fp"], at["fn"]) == (c["tp"][k], c["fp"][k], c["fn"][k])


def test_no_matching_files_raises():
    gt = FileAnnotations(id="0", path="a.png", height=10, width=10)
    pred = FileAnnotations(id="0", path="b.png", height=10, width=10)
    with pytest.raises(ValueError):
        evaluate(dataset([gt]), dataset([pred]))
