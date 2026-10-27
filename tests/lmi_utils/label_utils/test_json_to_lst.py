import json

import cv2
import numpy as np
import pytest

from lmi_utils.dataset_utils.representations import (
    Annotation,
    AnnotationType,
    Box,
    Dataset,
    FileAnnotations,
    Label,
    Mask,
    Point2d,
    Polygon,
    rle2mask,
)
from lmi_utils.label_utils.json_to_lst import dataset_to_lst, labeling_interface, to_image_root
from lmi_utils.label_utils.lst_brush import decode_mask, encode_mask
from lmi_utils.label_utils.lst_to_json import get_annotations_from_json

H, W = 100, 200


def random_mask(seed=0):
    mask = np.zeros((H, W), dtype=bool)
    mask[20:60, 30:90] = True
    mask[np.random.default_rng(seed).random((H, W)) > 0.97] = True
    return mask


def test_brush_rle_round_trip():
    mask = random_mask()
    assert np.array_equal(decode_mask(encode_mask(mask), H, W), mask)


def test_brush_rle_long_run_round_trip():
    # one run longer than the 16-bit run limit
    mask = np.zeros((300, 300), dtype=bool)
    mask[-1, -1] = True
    assert np.array_equal(decode_mask(encode_mask(mask), 300, 300), mask)


def test_brush_rle_matches_label_studio_sdk():
    brush = pytest.importorskip("label_studio_sdk.converter.brush")
    mask = random_mask(1)
    ours = encode_mask(mask)
    theirs = brush.mask2rle(mask.astype(np.uint8) * 255)
    assert np.array_equal(brush.decode_rle(ours).reshape(H, W, 4)[:, :, 3] > 128, mask)
    assert np.array_equal(decode_mask(theirs, H, W), mask)


def annot(i, kind, value, conf=None):
    return Annotation(id=str(i), label_id=kind, type=value_type(value), value=value, confidence=conf)


def value_type(value):
    return {
        Box: AnnotationType.BOX,
        Polygon: AnnotationType.POLYGON,
        Mask: AnnotationType.MASK,
        Point2d: AnnotationType.KEYPOINT,
    }[type(value)]


def build_dataset(mask):
    file = FileAnnotations(
        id="0",
        path="img0.png",
        height=H,
        width=W,
        annotations=[
            annot(0, "box", Box(x_min=10, y_min=20, x_max=60, y_max=80, angle=15)),
            annot(1, "poly", Polygon(points=[[10, 10], [100, 10], [100, 50]])),
            annot(2, "brush", Mask(mask=mask)),
            annot(3, "point", Point2d(x=40, y=30)),
        ],
        predictions=[annot(4, "box", Box(x_min=0, y_min=0, x_max=50, y_max=50, angle=0), conf=0.75)],
    )
    labels = [Label(id=k) for k in ("box", "poly", "brush", "point")]
    return Dataset(labels=labels, files=[file])


def test_json_to_lst_round_trips_through_lst_to_json(tmp_path):
    mask = random_mask()
    imgs = tmp_path / "imgs"
    imgs.mkdir()
    cv2.imwrite(str(imgs / "img0.png"), np.zeros((H, W, 3), dtype=np.uint8))
    lst_path = tmp_path / "lst.json"
    lst_path.write_text(json.dumps(dataset_to_lst(build_dataset(mask), "gs://bucket/imgs")))

    files, labels = get_annotations_from_json(str(lst_path), str(imgs))

    assert [f.path for f in files] == ["img0.png"]
    by_label = {a.label_id: a.value for a in files[0].annotations}
    box = by_label["box"]
    assert (box.x_min, box.y_min, box.x_max, box.y_max, box.angle) == (10, 20, 60, 80, 15)
    np.testing.assert_allclose(by_label["poly"].points, [[10, 10], [100, 10], [100, 50]], atol=1)
    assert np.array_equal(rle2mask(by_label["brush"].mask, h=H, w=W).astype(bool), mask)
    assert (by_label["point"].x, by_label["point"].y) == pytest.approx((40, 30))
    [pred] = files[0].predictions
    assert pred.label_id == "box" and pred.confidence == 0.75


def test_labeling_interface_lists_each_type():
    xml = labeling_interface(dataset_to_lst(build_dataset(random_mask()), "gs://b"))
    tags = ['RectangleLabels name="label"', 'PolygonLabels name="polygon"', 'BrushLabels name="brush"', 'KeyPointLabels name="keypoint"']
    for tag in tags:
        assert tag in xml


def test_local_image_root_becomes_local_files_url():
    assert to_image_root("/data/project/imgs") == "/data/local-files/?d=project/imgs"
    assert to_image_root("gs://bucket/imgs/") == "gs://bucket/imgs"
