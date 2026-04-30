import pytest

from object_detectors.od_core.od_base import ODBase


def test_none_operators_returns_empty_per_image():
    assert ODBase._normalize_operators(None, 3) == [[], [], []]


def test_single_chain_broadcasts_to_batch():
    chain = [{"resize": [640, 640, 1280, 720]}, {"pad": [0, 0, 80, 80]}]
    result = ODBase._normalize_operators(chain, 2)
    assert result == [chain, chain]


def test_per_image_chains_passthrough():
    chains = [[{"resize": [640, 640, 1280, 720]}], [{"pad": [0, 0, 80, 80]}]]
    assert ODBase._normalize_operators(chains, 2) == chains


def test_length_mismatch_raises():
    chains = [[{"resize": [640, 640, 1280, 720]}]]
    with pytest.raises(ValueError, match="must match batch size"):
        ODBase._normalize_operators(chains, 2)


def test_preprocessor_history_flat_chain_rejected():
    history = [
        {"type": "resize", "metadata": [{"orig_size": [1280, 720], "new_size": [640, 640]}]},
    ]
    with pytest.raises(ValueError, match="Preprocessor history"):
        ODBase._normalize_operators(history, 2)


def test_preprocessor_history_per_image_rejected():
    history_per_image = [
        [{"type": "resize", "metadata": [{"orig_size": [1280, 720], "new_size": [640, 640]}]}],
        [{"type": "tile", "metadata": [{"grid": [2, 2]}]}],
    ]
    with pytest.raises(ValueError, match="Preprocessor history"):
        ODBase._normalize_operators(history_per_image, 2)
