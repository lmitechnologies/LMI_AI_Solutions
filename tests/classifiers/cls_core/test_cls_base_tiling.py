"""ClassifierBase.predict() with tile operators, on a stub backend whose output depends on tile content."""

import numpy as np
import pytest

from classifiers.cls_core.cls_base import ClassifierBase
from lmi_utils.preprocess_utils import steps
from lmi_utils.preprocess_utils.ops import ResizeMeta, TileMeta
from lmi_utils.preprocess_utils.preprocessor import Preprocessor

NAMES = {0: "cat", 1: "dog", 2: "bird"}
# softmax row per tile pixel value: a tile filled with value v is classified as ROWS[v]
ROWS = {
    0: np.array([0.8, 0.15, 0.05], dtype=np.float32),
    1: np.array([0.05, 0.9, 0.05], dtype=np.float32),
    2: np.array([0.1, 0.1, 0.8], dtype=np.float32),
}


class StubClassifier(ClassifierBase):
    def __init__(self, class_map=NAMES):
        if class_map is not None:
            self._setup_class_map(class_map)
        self.postprocess_calls = []

    def warmup(self):
        pass

    def preprocess(self, images):
        return [int(np.asarray(im).flat[0]) for im in images]

    def forward(self, values):
        return values

    def postprocess(self, values, return_all_scores=False):
        self.postprocess_calls.append({"n": len(values), "return_all_scores": return_all_scores})
        rows = [ROWS[v] for v in values]
        out = {"classes": [NAMES[int(r.argmax())] for r in rows], "scores": [float(r.max()) for r in rows]}
        if return_all_scores:
            out["all_scores"] = [r.copy() for r in rows]
        return out


def _image(quadrants, tile=32):
    """2x2-tile image whose quadrants (row-major) are filled with the given values."""
    im = np.zeros((2 * tile, 2 * tile, 3), np.uint8)
    for k, v in enumerate(quadrants):
        r, c = divmod(k, 2)
        im[r * tile : (r + 1) * tile, c * tile : (c + 1) * tile] = v
    return im


def _tile(images, tile=32):
    return Preprocessor().preprocess(images, [steps.tile(tile_size=tile, stride=tile)])


def _tile_meta(n_tiles):
    n = len(n_tiles)
    return TileMeta(
        tile_sizes=[[32, 32]] * n,
        strides=[[32, 32]] * n,
        im_sizes=[[64, 64]] * n,
        scale_sizes=[[64, 64]] * n,
        n_tiles=n_tiles,
        batch_sizes=[1] * n,
        num_channels=[3] * n,
        scale_modes=["padding"] * n,
        overlap_modes=["average"] * n,
    )


# ── _setup_class_map ──────────────────────────────────────────────────────────


def test_setup_class_map_converts_string_keys():
    assert StubClassifier({"0": "a", "1": "b"}).class_map == {0: "a", 1: "b"}


def test_setup_class_map_raises_on_none():
    with pytest.raises(ValueError, match="required"):
        StubClassifier(class_map=None)._setup_class_map(None)


def test_setup_class_map_raises_on_non_dict():
    with pytest.raises(TypeError, match="dict"):
        StubClassifier(class_map=None)._setup_class_map(["cat", "dog"])


# ── without tiling ────────────────────────────────────────────────────────────


def test_predict_without_operators_is_per_image_and_plain_postprocess():
    stub = StubClassifier()
    out, time_info = stub.predict([np.full((8, 8, 3), v, np.uint8) for v in (0, 1, 2)])
    assert out["classes"] == ["cat", "dog", "bird"]
    assert "all_scores" not in out
    assert stub.postprocess_calls == [{"n": 3, "return_all_scores": False}]
    assert set(time_info) == {"preproc", "proc", "postproc"}


def test_non_tile_operators_do_not_merge():
    resize = ResizeMeta(src_sizes=[[128, 128]], dst_sizes=[[64, 64]], pads=[[0, 0, 0, 0]])
    out, _ = StubClassifier().predict([np.full((8, 8, 3), v, np.uint8) for v in (0, 1)], operators=[resize])
    assert out["classes"] == ["cat", "dog"]


def test_class_map_is_not_needed_without_tiling():
    out, _ = StubClassifier(class_map=None).predict(np.zeros((8, 8, 3), np.uint8))
    assert out["classes"] == ["cat"]


# ── tiled ─────────────────────────────────────────────────────────────────────


def test_tiled_predict_aggregates_tiles_by_content():
    # 3 dog tiles at .9 outweigh 1 cat tile at .8
    tiles, history = _tile([_image([1, 1, 0, 1])])
    stub = StubClassifier()
    out, _ = stub.predict(tiles, operators=history)
    assert out["classes"] == ["dog"]
    assert set(out) == {"classes", "scores"}
    assert stub.postprocess_calls[-1]["return_all_scores"] is True


def test_tiled_predict_keeps_source_images_apart():
    tiles, history = _tile([_image([0, 0, 0, 0]), _image([2, 2, 2, 2])])
    out, _ = StubClassifier().predict(tiles, operators=history)
    assert out["classes"] == ["cat", "bird"]
    assert out["scores"] == pytest.approx([0.8, 0.8], abs=1e-6)


@pytest.mark.parametrize("kwargs", [{"batch_size": 3}, {"batch_size": 1}])
def test_tiled_predict_is_the_same_in_mini_batches(kwargs):
    tiles, history = _tile([_image([1, 0, 0, 2]), _image([2, 1, 1, 1])])
    whole, _ = StubClassifier().predict(tiles, operators=history)
    stub = StubClassifier()
    chunked, _ = stub.predict(tiles, operators=history, **kwargs)
    assert chunked == whole
    assert all(c["return_all_scores"] for c in stub.postprocess_calls)


def test_tiled_predict_drops_fixed_batch_padding():
    # 8 tiles in chunks of 3 pad the last chunk with a zero image (a "cat" tile) that must not vote
    tiles, history = _tile([_image([1, 1, 1, 1]), _image([2, 2, 2, 2])])
    stub = StubClassifier()
    stub.fixed_batch_size = 3
    out, _ = stub.predict(tiles, operators=history)
    assert out["classes"] == ["dog", "bird"]
    assert out["scores"] == pytest.approx([0.9, 0.8], abs=1e-6)
    assert [c["n"] for c in stub.postprocess_calls] == [3, 3, 3]


def test_tile_count_mismatch_raises():
    with pytest.raises(ValueError, match="describes 4 tiles, but 3 images"):
        StubClassifier().predict([np.zeros((8, 8, 3), np.uint8)] * 3, operators=[_tile_meta([[2, 2]])])


def test_tiling_without_class_map_raises():
    with pytest.raises(RuntimeError, match="_setup_class_map"):
        StubClassifier(class_map=None).predict(np.zeros((8, 8, 3), np.uint8), operators=[_tile_meta([[1, 1]])])


# ── confidence filtering ─────────────────────────────────────────────────────


def test_confidence_zero_keeps_all():
    stub = StubClassifier()
    out, _ = stub.predict([np.full((8, 8, 3), v, np.uint8) for v in (0, 1, 2)], confidence=0)
    assert out["classes"] == ["cat", "dog", "bird"]
    assert out["scores"] == pytest.approx([0.8, 0.9, 0.8], abs=1e-6)


def test_confidence_threshold_filters_low_scores():
    # cat=0.8, dog=0.9, bird=0.8 → threshold 0.85 filters cat and bird
    stub = StubClassifier()
    out, _ = stub.predict([np.full((8, 8, 3), v, np.uint8) for v in (0, 1, 2)], confidence=0.85)
    assert out["classes"] == [None, "dog", None]
    assert out["scores"] == [None, pytest.approx(0.9, abs=1e-6), None]


def test_confidence_threshold_below_score_passes():
    # dog=0.9, threshold=0.89 → 0.9 < 0.89 is False, so dog passes
    stub = StubClassifier()
    out, _ = stub.predict([np.full((8, 8, 3), 1, np.uint8)], confidence=0.89)
    assert out["classes"] == ["dog"]
    assert out["scores"] == pytest.approx([0.9], abs=1e-6)


def test_confidence_threshold_filters_all():
    # threshold 0.95 > max score 0.9 → all filtered
    stub = StubClassifier()
    out, _ = stub.predict([np.full((8, 8, 3), v, np.uint8) for v in (0, 1, 2)], confidence=0.95)
    assert out["classes"] == [None, None, None]
    assert out["scores"] == [None, None, None]


def test_confidence_dict_per_class():
    # cat=0.8, dog=0.9, bird=0.8; thresholds: cat=0.5 (pass), dog=0.95 (fail), bird=0.5 (pass)
    stub = StubClassifier()
    out, _ = stub.predict(
        [np.full((8, 8, 3), v, np.uint8) for v in (0, 1, 2)],
        confidence={"cat": 0.5, "dog": 0.95, "bird": 0.5},
    )
    assert out["classes"] == ["cat", None, "bird"]
    assert out["scores"] == [pytest.approx(0.8, abs=1e-6), None, pytest.approx(0.8, abs=1e-6)]


def test_confidence_dict_unknown_class_defaults_to_one():
    # bird absent from the dict → threshold 1.0 (OD default) → always filtered
    stub = StubClassifier()
    out, _ = stub.predict(
        [np.full((8, 8, 3), v, np.uint8) for v in (0, 1, 2)],
        confidence={"cat": 0.5, "dog": 0.5},
    )
    assert out["classes"] == ["cat", "dog", None]
    assert out["scores"] == [pytest.approx(0.8, abs=1e-6), pytest.approx(0.9, abs=1e-6), None]


def test_confidence_none_raises():
    stub = StubClassifier()
    with pytest.raises(ValueError, match="not None"):
        stub.predict([np.full((8, 8, 3), 0, np.uint8)], confidence=None)


def test_confidence_filter_applies_after_tile_merge():
    # 3 dog tiles (0.9) + 1 cat tile (0.8) → merged dog at 0.7286; threshold 0.95 filters it
    tiles, history = _tile([_image([1, 1, 0, 1])])
    stub = StubClassifier()
    out, _ = stub.predict(tiles, operators=history, confidence=0.95)
    assert out["classes"] == [None]
    assert out["scores"] == [None]
