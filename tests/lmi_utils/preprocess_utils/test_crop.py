import numpy as np
import pytest
import torch

from lmi_utils.preprocess_utils.preprocessor import Preprocessor
from lmi_utils.preprocess_utils.reconstructor import Reconstructor


def _hwc_image(h, w, c=3):
    return torch.arange(h * w * c, dtype=torch.float32).reshape(h, w, c)


def test_crop_forward_basic():
    pre = Preprocessor()
    img = _hwc_image(100, 80, 3)
    steps = [{"type": "crop", "configuration": {"boxes": [[10, 20, 60, 90]]}}]
    out, history = pre.preprocess([img], steps)

    assert len(out) == 1
    assert out[0].shape == (70, 50, 3)
    assert history[0]["type"] == "crop"
    assert history[0]["metadata"][0]["box"] == [10, 20, 60, 90]
    assert history[0]["metadata"][0]["orig_shape"] == (100, 80)


def test_crop_clamps_out_of_bounds():
    pre = Preprocessor()
    img = _hwc_image(50, 50, 3)
    steps = [{"type": "crop", "configuration": {"boxes": [[-5, -5, 100, 100]]}}]
    out, history = pre.preprocess([img], steps)

    assert out[0].shape == (50, 50, 3)
    assert history[0]["metadata"][0]["box"] == [0, 0, 50, 50]


def test_crop_revert_image_pastes_into_canvas():
    pre, rec = Preprocessor(), Reconstructor()
    img = torch.ones((100, 80, 3), dtype=torch.float32)
    steps = [{"type": "crop", "configuration": {"boxes": [[10, 20, 60, 90]]}}]
    out, history = pre.preprocess([img], steps)

    restored = rec.reconstruct_images(out, history)
    assert restored[0].shape == (100, 80, 3)
    # Inside crop region: ones; outside: zeros.
    assert restored[0][20:90, 10:60].eq(1).all()
    assert restored[0][:20].eq(0).all()
    assert restored[0][90:].eq(0).all()
    assert restored[0][:, :10].eq(0).all()
    assert restored[0][:, 60:].eq(0).all()


def test_crop_revert_coords_adds_offset_to_boxes():
    pre, rec = Preprocessor(), Reconstructor()
    img = _hwc_image(100, 80, 3)
    steps = [{"type": "crop", "configuration": {"boxes": [[10, 20, 60, 90]]}}]
    _out, history = pre.preprocess([img], steps)

    # Box detected in crop-space, e.g. [5, 5, 15, 15] inside the 50x70 crop.
    results = {
        "boxes": [torch.tensor([[5.0, 5.0, 15.0, 15.0]])],
        "scores": [torch.tensor([0.9])],
        "classes": [np.array([0], dtype=np.int32)],
        "segments": [[]],
        "points": [torch.zeros((0, 1, 3))],
    }
    reverted = rec.reconstruct_coordinates(results, history)
    # Offset by (10, 20).
    assert torch.allclose(reverted["boxes"][0], torch.tensor([[15.0, 25.0, 25.0, 35.0]]))


def test_crop_revert_coords_masks_paste_into_full_canvas():
    pre, rec = Preprocessor(), Reconstructor()
    img = _hwc_image(100, 80, 3)
    steps = [{"type": "crop", "configuration": {"boxes": [[10, 20, 60, 90]]}}]
    _out, history = pre.preprocess([img], steps)

    # One mask of shape (1, 70, 50) — full crop.
    mask = torch.ones((1, 70, 50), dtype=torch.float32)
    results = {
        "boxes": [torch.zeros((0, 4))],
        "scores": [torch.zeros((0,))],
        "classes": [np.zeros((0,), dtype=np.int32)],
        "segments": [[]],
        "points": [torch.zeros((0, 1, 3))],
        "masks": [mask],
    }
    reverted = rec.reconstruct_coordinates(results, history)
    out_mask = reverted["masks"][0]
    assert out_mask.shape == (1, 100, 80)
    assert out_mask[0, 20:90, 10:60].eq(1).all()
    assert out_mask[0, :20].eq(0).all()


def test_crop_to_label_resolves_via_runtime():
    pre = Preprocessor()
    img = _hwc_image(100, 80, 3)
    steps = [{"type": "crop-to-label", "id": "label_crop", "configuration": {"label": "BOTTLE-BBOX"}}]
    runtime = {"label_crop": {"boxes": [[10, 20, 60, 90]]}}

    out, history = pre.preprocess([img], steps, runtime=runtime)
    assert out[0].shape == (70, 50, 3)
    # History records the resolved op name, not the macro name.
    assert history[0]["type"] == "crop"
    assert history[0]["id"] == "label_crop"


def test_crop_to_label_missing_runtime_raises():
    pre = Preprocessor()
    img = _hwc_image(100, 80, 3)
    steps = [{"type": "crop-to-label", "id": "label_crop", "configuration": {"label": "BOTTLE-BBOX"}}]
    with pytest.raises(ValueError, match="no runtime value"):
        pre.preprocess([img], steps, runtime=None)


def test_crop_to_label_then_resize_chain():
    pre, rec = Preprocessor(), Reconstructor()
    img = _hwc_image(100, 80, 3)
    steps = [
        {"type": "crop-to-label", "id": "label_crop", "configuration": {"label": "BOTTLE-BBOX"}},
        {"type": "resize", "configuration": {"width": 32, "height": 32, "preserve_aspect": False}},
    ]
    runtime = {"label_crop": {"boxes": [[10, 20, 60, 90]]}}
    out, history = pre.preprocess([img], steps, runtime=runtime)
    assert out[0].shape == (32, 32, 3)

    # Detected box in 32x32 resized space → revert through resize and crop.
    results = {
        "boxes": [torch.tensor([[0.0, 0.0, 32.0, 32.0]])],
        "scores": [torch.tensor([0.5])],
        "classes": [np.array([0], dtype=np.int32)],
        "segments": [[]],
        "points": [torch.zeros((0, 1, 3))],
    }
    reverted = rec.reconstruct_coordinates(results, history)
    # Resize reverts 32x32 → crop space (50x70); crop reverts by adding (10, 20).
    # Full-frame box in resized space should land at the crop's original location.
    box = reverted["boxes"][0][0].tolist()
    assert box == pytest.approx([10.0, 20.0, 60.0, 90.0], abs=1e-4)


def test_runtime_duplicate_ids_raises():
    pre = Preprocessor()
    img = _hwc_image(100, 80, 3)
    steps = [
        {"type": "crop-to-label", "id": "dup", "configuration": {"label": "A"}},
        {"type": "crop-to-label", "id": "dup", "configuration": {"label": "B"}},
    ]
    runtime = {"dup": {"boxes": [[0, 0, 10, 10]]}}
    with pytest.raises(ValueError, match="duplicate"):
        pre.preprocess([img], steps, runtime=runtime)


def test_runtime_id_routes_per_step():
    pre = Preprocessor()
    img = _hwc_image(100, 80, 3)
    steps = [
        {"type": "crop-to-label", "id": "a", "configuration": {"label": "A"}},
        {"type": "crop-to-label", "id": "b", "configuration": {"label": "B"}},
    ]
    runtime = {
        "a": {"boxes": [[0, 0, 10, 10]]},
        "b": {"boxes": [[5, 5, 25, 25]]},
    }
    out, history = pre.preprocess([img], steps, runtime=runtime)
    # Second crop runs on the output of the first; the first cropped to 10x10.
    # The second clamps [5, 5, 25, 25] against the 10x10 crop, yielding 5x5.
    assert out[0].shape == (5, 5, 3)
    assert history[0]["id"] == "a"
    assert history[1]["id"] == "b"


def test_runtime_unmatched_key_raises():
    pre = Preprocessor()
    img = _hwc_image(100, 80, 3)
    steps = [{"type": "resize", "configuration": {"width": 50, "height": 50}}]
    runtime = {"missing": {"boxes": [[0, 0, 10, 10]]}}
    with pytest.raises(ValueError, match="does not match"):
        pre.preprocess([img], steps, runtime=runtime)
