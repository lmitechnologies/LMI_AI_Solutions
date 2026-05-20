import numpy as np
import pytest
import torch

from lmi_utils.preprocess_utils.preprocessor import Preprocessor
from lmi_utils.preprocess_utils.reconstructor import Reconstructor


def _hwc_image(h, w, c=3):
    return torch.arange(h * w * c, dtype=torch.float32).reshape(h, w, c)


def _empty_results(n=1, **overrides):
    """Per-image results dict with empty defaults for every coord field.

    Pass `boxes=[...]`, `points=[...]`, etc. to override specific fields.
    Anything not overridden stays empty so handlers see no work for it.
    """
    results = {
        "boxes": [torch.zeros((0, 4)) for _ in range(n)],
        "scores": [torch.zeros((0,)) for _ in range(n)],
        "classes": [np.zeros((0,), dtype=np.int32) for _ in range(n)],
        "segments": [[] for _ in range(n)],
        "points": [torch.zeros((0, 1, 3)) for _ in range(n)],
    }
    results.update(overrides)
    return results


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
    results = _empty_results(boxes=[torch.tensor([[0.0, 0.0, 32.0, 32.0]])])
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


def test_crop_to_label_batch_per_image_runtime_boxes():
    """Runtime supplies one crop box per image; each image binds independently."""
    pre, rec = Preprocessor(), Reconstructor()
    img_a = _hwc_image(100, 80, 3)
    img_b = _hwc_image(120, 90, 3)
    steps = [{"type": "crop-to-label", "id": "lc", "configuration": {"label": "L"}}]
    runtime = {"lc": {"boxes": [[10, 20, 60, 90], [5, 5, 45, 65]]}}
    out, history = pre.preprocess([img_a, img_b], steps, runtime=runtime)
    assert out[0].shape == (70, 50, 3)
    assert out[1].shape == (60, 40, 3)

    # Verify per-image offsets propagate through revert.
    results = _empty_results(
        n=2,
        boxes=[torch.tensor([[0.0, 0.0, 10.0, 10.0]]), torch.tensor([[0.0, 0.0, 5.0, 5.0]])],
    )
    reverted = rec.reconstruct_coordinates(results, history)
    assert torch.allclose(reverted["boxes"][0], torch.tensor([[10.0, 20.0, 20.0, 30.0]]))
    assert torch.allclose(reverted["boxes"][1], torch.tensor([[5.0, 5.0, 10.0, 10.0]]))
