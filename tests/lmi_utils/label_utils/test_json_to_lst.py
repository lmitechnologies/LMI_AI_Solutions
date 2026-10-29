import json
import random

import cv2
import numpy as np
import pytest
from label_studio_sdk.converter import brush as ls_brush

from lmi_utils.dataset_utils.representations import (
    Annotation,
    AnnotationType,
    Box,
    BoxAnnotation,
    Dataset,
    FileAnnotations,
    KeypointAnnotation,
    Label,
    Mask,
    Point2d,
    Polygon,
    rle2mask,
)
from lmi_utils.label_utils.gadget_to_lst import convert_to_ls
from lmi_utils.label_utils.json_to_lst import dataset_to_lst
from lmi_utils.label_utils.lst_brush import _decode_values, decode_mask, encode_mask
from lmi_utils.label_utils.lst_project import labeling_interface, to_image_root
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


SDK_MASKS = {
    "noise": random_mask(1),
    "dense_noise": np.random.default_rng(2).random((H, W)) > 0.5,
    "empty": np.zeros((H, W), dtype=bool),
    "full": np.ones((H, W), dtype=bool),
    "long_run": np.pad(np.ones((1, 1), dtype=bool), ((299, 0), (299, 0))),
}


@pytest.mark.parametrize("name", SDK_MASKS)
def test_brush_rle_matches_label_studio_sdk(name):
    mask = SDK_MASKS[name]
    h, w = mask.shape
    rgba = np.repeat(mask.ravel().astype(np.uint8) * 255, 4)
    # the SDK decodes our encoding to the same RGBA bytes, and we decode the SDK's encoding to the same mask
    assert np.array_equal(ls_brush.decode_rle(encode_mask(mask)), rgba)
    assert np.array_equal(decode_mask(ls_brush.mask2rle(mask.astype(np.uint8) * 255), h, w), mask)


def random_rle_stream(rng):
    """A valid brush RLE with a random header and a random mix of run and multi-value literal blocks."""
    word_size = rng.randint(1, 8)
    rle_sizes = [rng.randint(1, 16) for _ in range(4)]
    num = rng.randint(0, 3000)
    parts = [f"{num:032b}", f"{word_size - 1:05b}"] + [f"{s - 1:04b}" for s in rle_sizes]
    values = []
    while len(values) < num:
        k = rng.randrange(4)
        n = min(rng.randint(1, 1 << rle_sizes[k]), num - len(values))
        is_run = rng.random() < 0.5
        block = [rng.randrange(1 << word_size)] * n if is_run else [rng.randrange(1 << word_size) for _ in range(n)]
        parts.append(f"{int(is_run)}{k:02b}{n - 1:0{rle_sizes[k]}b}")
        parts += [f"{v:0{word_size}b}" for v in (block[:1] if is_run else block)]
        values += block
    bits = "".join(parts)
    bits += "0" * (-len(bits) % 8)
    return [int(bits[i : i + 8], 2) for i in range(0, len(bits), 8)], np.array(values, dtype=np.uint8)


def test_brush_rle_decode_matches_label_studio_sdk_on_random_streams():
    # the SDK's encoder never writes multi-value literal blocks or non-default headers, but the LS frontend may
    rng = random.Random(0)
    for _ in range(500):
        rle, values = random_rle_stream(rng)
        assert np.array_equal(_decode_values(rle), values)
        assert np.array_equal(ls_brush.decode_rle(rle), values)


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
    lst_path.write_text(json.dumps(dataset_to_lst(build_dataset(mask), to_image_root("gs://bucket/imgs"))))

    # keypoints on rotated boxes are not linked
    files, labels = get_annotations_from_json(str(lst_path), str(imgs), keep_unlinked_keypoints=True)

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
    xml = labeling_interface(dataset_to_lst(build_dataset(random_mask()), to_image_root("gs://b")))
    tags = ['RectangleLabels name="label"', 'PolygonLabels name="polygon"', 'BrushLabels name="brush"', 'KeyPointLabels name="keypoint"']
    for tag in tags:
        assert tag in xml


def test_local_image_root_becomes_local_files_url():
    assert to_image_root("/data/project/imgs") == "/data/local-files/?d=project/imgs/"
    assert to_image_root("gs://bucket/imgs/") == "gs://bucket/imgs/"


def test_image_root_at_the_document_root_has_no_leading_slash():
    [task] = dataset_to_lst(build_dataset(random_mask()), to_image_root("/imgs"))
    assert task["data"]["image"] == "/data/local-files/?d=img0.png"


def test_gadget_to_lst_uses_json_to_lst_names(tmp_path):
    image = tmp_path / "folder" / "sensor" / "image" / "1.png"
    labels = tmp_path / "folder" / "pipeline" / "labels" / "1.label.json"
    for path in (image, labels):
        path.parent.mkdir(parents=True)
    image.write_bytes(b"")
    box = {"object": "dent", "x": 10, "y": 10, "width": 20, "height": 20, "score": 0.9}
    labels.write_text(json.dumps({"image_width": W, "image_height": H, "boxes": [box]}))

    convert_to_ls({"1": {"image": image, "labels": labels}}, str(tmp_path / "out"), "bucket/folder")

    [task] = json.loads((tmp_path / "out" / "label.json").read_text())
    assert task["predictions"][0]["result"][0]["from_name"] == "label"
    xml = (tmp_path / "out" / "labeling_interface.xml").read_text()
    assert '<RectangleLabels name="label" toName="image">' in xml and '<Label value="dent" />' in xml


def test_json_to_lst_round_trips_keypoint_links(tmp_path):
    imgs = tmp_path / "imgs"
    imgs.mkdir()
    cv2.imwrite(str(imgs / "img0.png"), np.zeros((H, W, 3), dtype=np.uint8))
    # the point lies in both boxes, so only the nesting can say which one owns it
    outer = BoxAnnotation(id="7", label_id="outer", value=Box(x_min=0, y_min=0, x_max=150, y_max=90))
    inner = BoxAnnotation(id="8", label_id="inner", value=Box(x_min=20, y_min=20, x_max=80, y_max=80))
    point = KeypointAnnotation(id="9", label_id="tip", value=Point2d(x=50, y=50), bounding_box_id="8")
    file = FileAnnotations(id="0", path="img0.png", height=H, width=W, annotations=[outer, inner, point])
    lst_path = tmp_path / "lst.json"
    lst_path.write_text(json.dumps(dataset_to_lst(Dataset(labels=[], files=[file]), to_image_root("gs://bucket/imgs"))))

    [converted], _ = get_annotations_from_json(str(lst_path), str(imgs))

    by_label = {a.label_id: a for a in converted.annotations}
    assert by_label["tip"].bounding_box_id == by_label["inner"].id
