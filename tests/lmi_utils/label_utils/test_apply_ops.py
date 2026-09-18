import json

import cv2
import numpy as np
import pytest

from lmi_utils.label_utils.apply_ops import apply_ops


def _write_dataset(path_in, annotations, h=100, w=150):
    path_in.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path_in / "a.png"), np.full((h, w, 3), 7, np.uint8))
    labels = {
        "labels": [{"id": "defect", "color": "#ff0000", "annotation_type": "Box"}],
        "files": [{"id": "1", "path": "a.png", "height": h, "width": w, "annotations": annotations, "predictions": []}],
    }
    (path_in / "labels.json").write_text(json.dumps(labels))
    return path_in


def _box(id_, x1, y1, x2, y2):
    return {
        "id": id_,
        "label_id": "defect",
        "type": "Box",
        "value": {"x_min": x1, "y_min": y1, "x_max": x2, "y_max": y2, "angle": 0},
    }


def _tile_args(path_in, path_out, **overrides):
    args = {
        "operation": "tile",
        "path_imgs": str(path_in),
        "path_json": "labels.json",
        "path_out_images": str(path_out),
        "path_out_json": "labels.json",
        "bg": False,
        "warn_crop": False,
        "width": 64,
        "height": 64,
        "stride": [50],
        "min_label_size": 0.0,
    }
    args.update(overrides)
    return args


def _run(tmp_path, annotations, **overrides):
    path_in, path_out = _write_dataset(tmp_path / "in", annotations), tmp_path / "out"
    apply_ops(_tile_args(path_in, path_out, **overrides))
    return path_out, json.loads((path_out / "labels.json").read_text())


def test_tile_writes_images_and_labels_that_agree(tmp_path):
    path_out, out = _run(tmp_path, [_box("b1", 40, 10, 90, 60)])

    # a 100x150 image at tile 64 stride 50 is a 2x3 grid; the two tiles right of the box hold no labels
    assert len(out["files"]) == 4
    for f in out["files"]:
        image = cv2.imread(str(path_out / f["path"]))
        assert image is not None, f"labels.json points at a missing image: {f['path']}"
        assert (image.shape[0], image.shape[1]) == (f["height"], f["width"]) == (64, 64)
        for annot in f["annotations"]:
            v = annot["value"]
            assert 0 <= v["x_min"] < v["x_max"] <= f["width"]
            assert 0 <= v["y_min"] < v["y_max"] <= f["height"]


def test_tile_records_the_source_file_and_a_position_free_of_repeats(tmp_path):
    _, out = _run(tmp_path, [_box("b1", 40, 10, 90, 60)])

    assert {f["source_id"] for f in out["files"]} == {"1"}, "every tile points back at the one source image"
    assert [f["id"] for f in out["files"]] == ["1_r0_c0", "1_r0_c1", "1_r1_c0", "1_r1_c1"]
    assert [f["path"] for f in out["files"]] == [f"id1_r{r}_c{c}_a_tile_64x64.png" for r in (0, 1) for c in (0, 1)]


def test_tile_reassembles_to_the_source_box(tmp_path):
    """Each tile holds a piece of the box; shifted back by its origin they cover the original."""
    _, out = _run(tmp_path, [_box("b1", 40, 10, 90, 60)])

    by_id = {f["id"]: f for f in out["files"]}
    corners = []
    for row in (0, 1):
        for col in (0, 1):
            for annot in by_id[f"1_r{row}_c{col}"]["annotations"]:
                v = annot["value"]
                corners.append([v["x_min"] + col * 50, v["y_min"] + row * 50, v["x_max"] + col * 50, v["y_max"] + row * 50])

    corners = np.array(corners)
    assert [corners[:, 0].min(), corners[:, 1].min(), corners[:, 2].max(), corners[:, 3].max()] == [40, 10, 90, 60]


def test_tile_drops_empty_tiles_unless_bg_is_set(tmp_path):
    _, without_bg = _run(tmp_path / "no_bg", [_box("b1", 0, 0, 20, 20)])
    assert [f["id"] for f in without_bg["files"]] == ["1_r0_c0"]

    _, with_bg = _run(tmp_path / "bg", [_box("b1", 0, 0, 20, 20)], bg=True)
    assert len(with_bg["files"]) == 6, "the full 2x3 grid survives when background tiles are kept"
    assert sum(len(f["annotations"]) for f in with_bg["files"]) == 1


def test_tile_rejects_an_oriented_box(tmp_path):
    obb = _box("b1", 40, 10, 90, 60)
    obb["value"]["angle"] = 30

    with pytest.raises(ValueError, match="does not support oriented boxes"):
        _run(tmp_path, [obb])


def test_tile_requires_both_dimensions(tmp_path):
    with pytest.raises(ValueError, match="requires both --width and --height"):
        _run(tmp_path, [_box("b1", 40, 10, 90, 60)], width=None)
