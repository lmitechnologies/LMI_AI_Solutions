import numpy as np
import pytest
import torch

from lmi_utils.preprocess_utils import steps
from lmi_utils.preprocess_utils.ops import CropMeta
from lmi_utils.preprocess_utils.preprocessor import Preprocessor
from lmi_utils.preprocess_utils.reconstructor import Reconstructor


def _hwc_image(h, w, c=3):
    return torch.arange(h * w * c, dtype=torch.float32).reshape(h, w, c)


def _empty_results(n=1, **overrides):
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
    configs = [steps.crop_to_label(label="BOTTLE-BBOX", id="label_crop")]
    runtime = {"BOTTLE-BBOX": {"boxes": [[10, 20, 60, 90]]}}

    out, history = pre.preprocess([img], configs, runtime=runtime)
    assert out[0].shape == (70, 50, 3)
    # History records the resolved op's Meta type.
    assert isinstance(history[0], CropMeta)


def test_crop_to_label_missing_runtime_raises():
    pre = Preprocessor()
    img = _hwc_image(100, 80, 3)
    configs = [steps.crop_to_label(label="BOTTLE-BBOX", id="label_crop")]
    with pytest.raises(ValueError, match="no runtime value"):
        pre.preprocess([img], configs, runtime=None)


def test_crop_to_label_then_resize_chain():
    pre, rec = Preprocessor(), Reconstructor()
    img = _hwc_image(100, 80, 3)
    configs = [
        steps.crop_to_label(label="BOTTLE-BBOX", id="label_crop"),
        steps.resize(width=32, height=32, preserve_aspect=False),
    ]
    runtime = {"BOTTLE-BBOX": {"boxes": [[10, 20, 60, 90]]}}
    out, history = pre.preprocess([img], configs, runtime=runtime)
    assert out[0].shape == (32, 32, 3)

    results = _empty_results(boxes=[torch.tensor([[0.0, 0.0, 32.0, 32.0]])])
    reverted = rec.reconstruct_coordinates(results, history)
    box = reverted["boxes"][0][0].tolist()
    assert box == pytest.approx([10.0, 20.0, 60.0, 90.0], abs=1e-4)


def test_runtime_duplicate_labels_raises():
    pre = Preprocessor()
    img = _hwc_image(100, 80, 3)
    configs = [
        steps.crop_to_label(label="dup", id="a"),
        steps.crop_to_label(label="dup", id="b"),
    ]
    runtime = {"dup": {"boxes": [[0, 0, 10, 10]]}}
    with pytest.raises(ValueError, match="Duplicate crop-to-label label"):
        pre.preprocess([img], configs, runtime=runtime)


def test_runtime_routes_per_label():
    pre = Preprocessor()
    img = _hwc_image(100, 80, 3)
    configs = [
        steps.crop_to_label(label="A", id="a"),
        steps.crop_to_label(label="B", id="b"),
    ]
    runtime = {
        "A": {"boxes": [[0, 0, 10, 10]]},
        "B": {"boxes": [[5, 5, 25, 25]]},
    }
    out, history = pre.preprocess([img], configs, runtime=runtime)
    assert out[0].shape == (5, 5, 3)
    assert isinstance(history[0], CropMeta)
    assert isinstance(history[1], CropMeta)


def test_runtime_unmatched_label_raises():
    pre = Preprocessor()
    img = _hwc_image(100, 80, 3)
    configs = [steps.crop_to_label(label="BOTTLE-BBOX", id="a")]
    runtime = {"WRONG-LABEL": {"boxes": [[0, 0, 10, 10]]}}
    with pytest.raises(ValueError, match="does not match"):
        pre.preprocess([img], configs, runtime=runtime)


def test_runtime_unmatched_key_raises():
    pre = Preprocessor()
    img = _hwc_image(100, 80, 3)
    configs = [steps.resize(width=50, height=50)]
    runtime = {"missing": {"boxes": [[0, 0, 10, 10]]}}
    with pytest.raises(ValueError, match="does not match"):
        pre.preprocess([img], configs, runtime=runtime)


def test_crop_to_label_batch_per_image_runtime_boxes():
    pre, rec = Preprocessor(), Reconstructor()
    img_a = _hwc_image(100, 80, 3)
    img_b = _hwc_image(120, 90, 3)
    configs = [steps.crop_to_label(label="L", id="lc")]
    runtime = {"L": {"boxes": [[10, 20, 60, 90], [5, 5, 45, 65]]}}
    out, history = pre.preprocess([img_a, img_b], configs, runtime=runtime)
    assert out[0].shape == (70, 50, 3)
    assert out[1].shape == (60, 40, 3)

    results = _empty_results(
        n=2,
        boxes=[torch.tensor([[0.0, 0.0, 10.0, 10.0]]), torch.tensor([[0.0, 0.0, 5.0, 5.0]])],
    )
    reverted = rec.reconstruct_coordinates(results, history)
    assert torch.allclose(reverted["boxes"][0], torch.tensor([[10.0, 20.0, 20.0, 30.0]]))
    assert torch.allclose(reverted["boxes"][1], torch.tensor([[5.0, 5.0, 10.0, 10.0]]))
