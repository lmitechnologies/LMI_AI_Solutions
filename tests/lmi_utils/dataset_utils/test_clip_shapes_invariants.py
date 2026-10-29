"""Randomized checks of what clip_shapes must hold for any input, edge-touching and fractional coordinates included."""

import copy
import os

import numpy as np
import pytest
import shapely
from shapely.geometry import Point as ShapelyPoint
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.geometry import box as shapely_box

from lmi_utils.dataset_utils.ops.dataset_pad import clip_shapes
from lmi_utils.dataset_utils.representations import (
    AnnotationType,
    Box,
    BoxAnnotation,
    KeypointAnnotation,
    Mask,
    MaskAnnotation,
    Point2d,
    Polygon,
    PolygonAnnotation,
)

# CLIP_SHAPES_SEED_START and CLIP_SHAPES_SEEDS run a longer or different search
_START = int(os.environ.get("CLIP_SHAPES_SEED_START", "0"))
SEEDS = range(_START, _START + int(os.environ.get("CLIP_SHAPES_SEEDS", "50")))
CASES_PER_SEED = 40
TOL = 1e-6


def _coord(rng, size):
    # often exactly on an edge or a whole pixel, where equality checks go wrong
    r = rng.random()
    if r < 0.15:
        return float(rng.choice([0, size]))
    if r < 0.3:
        return float(rng.integers(-size // 2, size + size // 2))
    return float(rng.uniform(-0.5 * size, 1.5 * size))


def _upright(rng, W, H):
    x1, x2 = sorted([_coord(rng, W), _coord(rng, W)])
    y1, y2 = sorted([_coord(rng, H), _coord(rng, H)])
    return Box(x1, y1, x2 + 1, y2 + 1)


def _rotated(rng, W, H):
    x, y = _coord(rng, W), _coord(rng, H)
    return Box(x, y, x + rng.uniform(2, W), y + rng.uniform(2, H), float(rng.choice([rng.uniform(-89, 89), 45, -30])))


def _polygon(rng, W, H):
    n = int(rng.integers(3, 12))
    cx, cy = _coord(rng, W), _coord(rng, H)
    angles = np.sort(rng.uniform(0, 2 * np.pi, n))
    if rng.random() < 0.15:
        rng.shuffle(angles)  # a self-crossing outline
    radii = rng.uniform(1, max(W, H) / 2, n)
    points = np.stack([cx + radii * np.cos(angles), cy + radii * np.sin(angles)], axis=1)
    if rng.random() < 0.2:
        points = np.round(points)
    return Polygon(points=points.tolist())


def _mask(rng, W, H):
    mask = np.zeros((H, W), dtype=np.uint8)
    if rng.random() < 0.8:
        x, y = int(rng.integers(0, W)), int(rng.integers(0, H))
        mask[y : y + int(rng.integers(1, 10)), x : x + int(rng.integers(1, 10))] = 1
    return Mask(mask=mask)


def _case(rng):
    W, H = int(rng.integers(20, 200)), int(rng.integers(20, 200))
    shapes = []
    for i in range(int(rng.integers(1, 8))):
        kind = rng.choice(["box", "rbox", "polygon", "mask"])
        if kind == "box":
            shapes.append(BoxAnnotation(f"b{i}", "part", _upright(rng, W, H)))
        elif kind == "rbox":
            shapes.append(BoxAnnotation(f"r{i}", "part", _rotated(rng, W, H)))
        elif kind == "polygon":
            shapes.append(PolygonAnnotation(f"p{i}", "dent", _polygon(rng, W, H)))
        else:
            shapes.append(MaskAnnotation(f"m{i}", "dent", _mask(rng, W, H)))
    boxes = [s for s in shapes if s.type == AnnotationType.BOX]
    for i in range(int(rng.integers(0, 6))):
        box = boxes[int(rng.integers(len(boxes)))] if boxes and rng.random() < 0.7 else None
        if box is not None and rng.random() < 0.8:
            # inside its own box, as a correct label is
            outline = ShapelyPolygon(box.value.to_polygon().points)
            x1, y1, x2, y2 = outline.bounds
            while True:
                x, y = rng.uniform(x1, x2), rng.uniform(y1, y2)
                if outline.covers(ShapelyPoint(x, y)):
                    break
        else:
            x, y = _coord(rng, W), _coord(rng, H)
        shapes.append(KeypointAnnotation(f"k{i}", "tip", Point2d(x, y), bounding_box_id=box.id if box else None))
    return W, H, shapes


def _filled_area(geom):
    # an AIS polygon is one outline, so a hole make_valid finds in a self-crossing one is filled
    if isinstance(geom, ShapelyPolygon):
        return ShapelyPolygon(geom.exterior).area
    return sum(_filled_area(g) for g in getattr(geom, "geoms", []))


def _outline(value):
    return ShapelyPolygon(value.to_polygon().points if isinstance(value, Box) else value.points)


def _check(W, H, shapes):
    values = [s.value for s in shapes]
    before = copy.deepcopy(shapes)
    kept, changed = clip_shapes(shapes, W, H)
    image = shapely_box(0, 0, W, H)
    by_id = {s.id: s for s in kept}
    any_change = False

    for orig, value in zip(before, values):
        out = [s for s in kept if s.id == orig.id or s.id.startswith(f"{orig.id}_p")]
        v = orig.value
        if orig.type == AnnotationType.BOX and not v.angle:
            expected = np.clip(v.to_numpy()[:-1], 0, [W, H, W, H])
            if expected[0] == expected[2] or expected[1] == expected[3]:
                assert out == [], f"{orig.id} has no area inside the image but was kept"
                any_change = True
                continue
            [box] = out
            np.testing.assert_allclose(box.value.to_numpy()[:-1], expected)
            any_change |= not np.array_equal(expected, v.to_numpy()[:-1])

        elif orig.type == AnnotationType.BOX:
            part = _outline(v).intersection(image)
            if not part.area:
                assert out == [], f"{orig.id} has no area inside the image but was kept"
                any_change = True
                continue
            [box] = out
            assert box.value.angle == v.angle
            assert _outline(box.value).buffer(TOL).covers(part), f"{orig.id} no longer covers its part inside the image"
            assert box.value.area() <= v.area() + TOL
            any_change |= not image.covers(_outline(v))

        elif orig.type == AnnotationType.POLYGON:
            valid = shapely.make_valid(_outline(v))
            area_inside = _filled_area(valid.intersection(image))
            if not area_inside:
                assert out == [], f"{orig.id} has no area inside the image but was kept"
                any_change |= bool(v.points)
                continue
            for piece in out:
                points = np.asarray(piece.value.points)
                assert points.min() >= -TOL and points[:, 0].max() <= W + TOL and points[:, 1].max() <= H + TOL
            if len(out) == 1 and out[0].value is value:
                points = np.asarray(v.points)
                assert points.min() >= 0 and points[:, 0].max() <= W and points[:, 1].max() <= H
            else:
                assert sum(_outline(p.value).area for p in out) <= _filled_area(valid) + TOL
                assert sum(_outline(p.value).area for p in out) == pytest.approx(area_inside, rel=1e-6, abs=1e-6)
                any_change = True

        elif orig.type == AnnotationType.MASK:
            if v.to_numpy(h=H, w=W).any():
                assert len(out) == 1
            else:
                assert out == []
                any_change = True

        elif orig.type == AnnotationType.KEYPOINT:
            x, y = v.x, v.y
            box_removed = orig.bounding_box_id is not None and orig.bounding_box_id not in by_id
            if not (0 <= x <= W and 0 <= y <= H) or box_removed:
                assert out == [], f"{orig.id} at ({x}, {y}) should be dropped"
                any_change = True
                continue
            [kp] = out
            assert (kp.value.x, kp.value.y) == (x, y) and kp.bounding_box_id == orig.bounding_box_id
            box = next((b for b in before if b.id == orig.bounding_box_id), None)
            if box is not None and _outline(box.value).buffer(TOL).covers(ShapelyPoint(x, y)):
                assert _outline(by_id[box.id].value).buffer(TOL).covers(ShapelyPoint(x, y)), f"{orig.id} left its box"

    assert changed == any_change


@pytest.mark.parametrize("seed", SEEDS)
def test_clip_shapes_invariants(seed):
    rng = np.random.default_rng(seed)
    for _ in range(CASES_PER_SEED):
        _check(*_case(rng))
