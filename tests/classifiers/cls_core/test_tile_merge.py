"""Unit tests for ClassifierBase._merge_tile_predictions. No model or GPU needed."""

import numpy as np
import pytest

from classifiers.cls_core.cls_base import ClassifierBase
from lmi_utils.preprocess_utils.ops import TileMeta


def _tile_meta(n_tiles):
    n = len(n_tiles)
    return TileMeta(
        tile_sizes=[[224, 224]] * n,
        strides=[[224, 224]] * n,
        im_sizes=[[448, 448]] * n,
        scale_sizes=[[448, 448]] * n,
        n_tiles=n_tiles,
        batch_sizes=[1] * n,
        num_channels=[3] * n,
        scale_modes=["padding"] * n,
        overlap_modes=["average"] * n,
    )


class _StubMerger(ClassifierBase):
    def warmup(self):
        pass

    def preprocess(self, images):
        pass

    def forward(self, x):
        pass

    def postprocess(self, x):
        pass


@pytest.fixture
def merger():
    m = _StubMerger()
    m._setup_class_map({0: "cat", 1: "dog", 2: "bird"})
    return m


def _merge(merger, vecs, n_tiles):
    tile_results = {"all_scores": [np.array(v, dtype=np.float32) for v in vecs]}
    return merger._merge_tile_predictions(tile_results, _tile_meta(n_tiles))


def test_all_tiles_agree_returns_that_class(merger):
    out = _merge(merger, [[0.9, 0.08, 0.02], [0.85, 0.1, 0.05]], [[1, 2]])
    assert out["classes"] == ["cat"]


def test_confidence_is_weighted_fraction_of_total_mass(merger):
    # (0.9*0.9 + 0.85*0.85) / (0.9 + 0.85)
    out = _merge(merger, [[0.9, 0.08, 0.02], [0.85, 0.1, 0.05]], [[1, 2]])
    assert out["scores"][0] == pytest.approx(1.5325 / 1.75, abs=1e-6)


def test_a_confident_tile_outweighs_a_weak_majority(merger):
    # two weak dog votes vs one confident cat: cat 1.3325 > dog 0.691
    out = _merge(merger, [[0.4, 0.55, 0.05], [0.35, 0.6, 0.05], [0.95, 0.03, 0.02]], [[1, 3]])
    assert out["classes"] == ["cat"]


def test_winner_that_is_no_tiles_top_class_gets_its_own_name(merger):
    # dog is second in both tiles but has the most mass: dog .4896 > cat = bird .2652
    out = _merge(merger, [[0.51, 0.48, 0.01], [0.01, 0.48, 0.51]], [[1, 2]])
    assert out["classes"] == ["dog"]
    assert out["scores"][0] == pytest.approx(0.48, abs=1e-6)


def test_confidence_in_zero_one(merger):
    out = _merge(merger, [[0.1, 0.8, 0.1], [0.2, 0.7, 0.1], [0.6, 0.2, 0.2]], [[1, 3]])
    assert 0.0 <= out["scores"][0] <= 1.0


def test_one_by_one_grid_returns_the_tiles_own_prediction(merger):
    out = _merge(merger, [[0.1, 0.05, 0.85]], [[1, 1]])
    assert out["classes"] == ["bird"]
    assert out["scores"][0] == pytest.approx(0.85, abs=1e-6)


def test_multiple_source_images_split_at_their_tile_counts(merger):
    # an off-by-one split would leak a dog tile into the first image and flip it
    cat, dog = [0.8, 0.15, 0.05], [0.0, 1.0, 0.0]
    out = _merge(merger, [cat] * 4 + [dog] * 3, [[2, 2], [1, 3]])
    assert out["classes"] == ["cat", "dog"]
    assert out["scores"] == pytest.approx([0.8, 1.0], abs=1e-6)


def test_output_has_only_classes_and_scores_keys(merger):
    out = _merge(merger, [[0.7, 0.2, 0.1]], [[1, 1]])
    assert set(out) == {"classes", "scores"}
