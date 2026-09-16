"""The chain the tiling work exists to serve: labels.json -> apply_ops tile -> json_to_yolo -> train."""

import json

import cv2
import numpy as np
import pytest

from lmi_utils.dataset_utils.mask_encoder import mask2rle
from lmi_utils.label_utils.apply_ops import apply_ops
from lmi_utils.label_utils.json_to_yolo import convert_to_yolo

TILE, STRIDE = 64, 50
SOURCE_H, SOURCE_W = 100, 150


def _annotation(kind):
    if kind == "Box":
        return {"id": "a1", "label_id": "defect", "type": "Box", "value": {"x_min": 40, "y_min": 10, "x_max": 90, "y_max": 60, "angle": 0}}
    if kind == "Polygon":
        return {"id": "a1", "label_id": "defect", "type": "Polygon", "value": {"points": [[40, 10], [90, 10], [90, 60], [40, 60]]}}
    mask = np.zeros((SOURCE_H, SOURCE_W), np.uint8)
    mask[10:60, 40:90] = 1
    return {"id": "a1", "label_id": "defect", "type": "Bitmask", "value": {"mask": mask2rle(mask)}}


def _write_source(path_in, kind):
    path_in.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path_in / "a.png"), np.full((SOURCE_H, SOURCE_W, 3), 7, np.uint8))
    (path_in / "labels.json").write_text(
        json.dumps(
            {
                "labels": [{"id": "defect", "color": "#ff0000", "annotation_type": kind}],
                "files": [
                    {
                        "id": "1",
                        "path": "a.png",
                        "height": SOURCE_H,
                        "width": SOURCE_W,
                        "annotations": [_annotation(kind)],
                        "predictions": [],
                    }
                ],
            }
        )
    )
    return path_in


def _tile_then_yolo(tmp_path, kind, seg=False):
    path_in, path_tiles, path_yolo = _write_source(tmp_path / "in", kind), tmp_path / "tiles", tmp_path / "yolo"
    apply_ops(
        {
            "operation": "tile",
            "path_imgs": str(path_in),
            "path_json": "labels.json",
            "path_out_images": str(path_tiles),
            "path_out_json": "labels.json",
            "bg": False,
            "warn_crop": False,
            "width": TILE,
            "height": TILE,
            "stride": [STRIDE],
            "min_label_size": 0.0,
        }
    )
    path_yolo.mkdir()
    convert_to_yolo(
        {
            "path_train_imgs": str(path_tiles),
            "path_val_imgs": None,
            "path_train_json": "labels.json",
            "path_val_json": "labels.json",
            "path_out": str(path_yolo),
            "target_classes": "defect",
            "seg": seg,
            "obb": False,
            "bg": False,
            "merge_box": False,
        }
    )
    return path_tiles, path_yolo


def _rows(path_yolo):
    out = {}
    for txt in sorted((path_yolo / "labels/train").glob("*.txt")):
        out[txt.stem] = [line.split() for line in txt.read_text().splitlines() if line.strip()]
    return out


@pytest.mark.parametrize("kind", ["Box", "Polygon", "Bitmask"])
def test_tiled_dataset_converts_to_yolo(tmp_path, kind):
    path_tiles, path_yolo = _tile_then_yolo(tmp_path, kind, seg=kind != "Box")

    images = sorted(p.stem for p in (path_yolo / "images/train").glob("*.png"))
    rows = _rows(path_yolo)

    assert images, "no tile images reached the yolo dataset"
    assert sorted(rows) == images, "every tile image needs a label file and vice versa"
    for name, lines in rows.items():
        assert lines, f"{name} lost its label in conversion"
        for line in lines:
            assert line[0] == "0", "the one target class indexes to 0"
            coords = np.array(line[1:], dtype=float)
            assert ((coords >= 0) & (coords <= 1)).all(), f"{name} has coordinates outside the tile"


def test_the_id_prefix_is_not_applied_twice_along_the_chain(tmp_path):
    """apply_ops, json_to_yolo and plot_with_json all prepend id<id>_ unless it is already there."""
    _, path_yolo = _tile_then_yolo(tmp_path, "Box")

    names = sorted(p.name for p in (path_yolo / "images/train").glob("*.png"))
    assert names == [f"id1_r{r}_c{c}_a_tile_{TILE}x{TILE}.png" for r in (0, 1) for c in (0, 1)]
    for name in names:
        assert name.count("_r0_c") + name.count("_r1_c") == 1, f"tile position repeated in {name}"


def test_tiled_boxes_cover_the_source_box(tmp_path):
    """Un-normalize each tile's yolo row, shift by the tile origin, and the union is the source box."""
    _, path_yolo = _tile_then_yolo(tmp_path, "Box")

    corners = []
    for name, lines in _rows(path_yolo).items():
        row, col = int(name.split("_r")[1][0]), int(name.split("_c")[1][0])
        for _, cx, cy, w, h in lines:
            cx, cy, w, h = float(cx) * TILE, float(cy) * TILE, float(w) * TILE, float(h) * TILE
            corners.append([cx - w / 2 + col * STRIDE, cy - h / 2 + row * STRIDE, cx + w / 2 + col * STRIDE, cy + h / 2 + row * STRIDE])

    corners = np.array(corners)
    bounds = [corners[:, 0].min(), corners[:, 1].min(), corners[:, 2].max(), corners[:, 3].max()]
    # the yolo txt stores rounded normalized coords, so the pixel round-trip is not exact
    assert bounds == pytest.approx([40, 10, 90, 60], abs=0.05)


def test_segmentation_outlines_stay_inside_their_tile(tmp_path):
    _, path_yolo = _tile_then_yolo(tmp_path, "Polygon", seg=True)

    for name, lines in _rows(path_yolo).items():
        for line in lines:
            points = np.array(line[1:], dtype=float).reshape(-1, 2)
            assert len(points) >= 3, f"{name} produced an outline with too few points"
            assert ((points >= 0) & (points <= 1)).all()
