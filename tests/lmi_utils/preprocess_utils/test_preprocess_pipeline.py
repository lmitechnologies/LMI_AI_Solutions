import numpy as np
import pytest
import torch
from preprocess_utils.preprocessor import Preprocessor
from preprocess_utils.reconstructor import Reconstructor


@pytest.fixture
def pipeline():
    """
    Initializes the real Preprocessor and Reconstructor.
    """
    # Initialize classes
    prep = Preprocessor()
    recon = Reconstructor()

    return prep, recon


@pytest.mark.parametrize("input_type", ["numpy", "torch"])
def test_tiling_lossless_reconstruction(pipeline, input_type):
    prep, recon = pipeline

    if input_type == "torch":
        input_image = torch.randint(0, 256, (100, 100, 3), dtype=torch.uint8)
    else:
        input_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

    steps = [{"type": "tile", "configuration": {"tile_size": 50, "stride": 50}}]

    final_images, ops = prep.preprocess(input_image, steps)

    assert len(final_images) == 4
    assert final_images[0].shape == (50, 50, 3)

    restored_image = recon.reconstruct(final_images, ops)

    assert isinstance(restored_image, type(input_image))
    if input_type == "torch":
        assert restored_image.shape == input_image.shape
        assert torch.allclose(restored_image.float(), input_image.float())
    else:
        assert restored_image.shape == input_image.shape
        assert np.allclose(restored_image, input_image)


@pytest.mark.parametrize("input_type", ["numpy", "torch"])
def test_nested_pipeline_flow(pipeline, input_type):
    prep, recon = pipeline

    if input_type == "torch":
        input_image = torch.randint(0, 256, (128, 128, 3), dtype=torch.uint8)
    else:
        input_image = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)

    steps = [
        {"type": "resize", "configuration": {"width": 64, "height": 64}},
        {"type": "tile", "configuration": {"tile_size": 32, "stride": 32}},
        {"type": "tile", "configuration": {"tile_size": 16, "stride": 16}},
    ]

    final_images, ops = prep.preprocess(input_image, steps)

    assert len(final_images) == 16
    assert final_images[0].shape == (16, 16, 3)

    restored_image = recon.reconstruct(final_images, ops)

    assert isinstance(restored_image, type(input_image))
    if input_type == "torch":
        assert restored_image.shape == (128, 128, 3)


@pytest.mark.parametrize(
    "shape, steps, expected_final_count, expected_final_shape, seed",
    [
        # Simple case: 77x64 -> 32x32 -> 16x16
        (
            (77, 64, 3),
            [
                {"type": "tile", "configuration": {"tile_size": 32, "stride": 32}},
                {"type": "tile", "configuration": {"tile_size": 16, "stride": 16}},
            ],
            24,
            (16, 16, 3),
            42,
        ),
        # Case with overlap: 100x100 -> 50x50 -> 25x25
        (
            (100, 100, 3),
            [
                {"type": "tile", "configuration": {"tile_size": 50, "stride": 25}},
                {"type": "tile", "configuration": {"tile_size": 25, "stride": 25}},
            ],
            36,
            (25, 25, 3),
            43,
        ),
        # Non-square: 128x64 -> 64x32 -> 32x16
        (
            (128, 64, 3),
            [
                {"type": "tile", "configuration": {"tile_size": [64, 32], "stride": [64, 32]}},
                {"type": "tile", "configuration": {"tile_size": [32, 16], "stride": [32, 16]}},
            ],
            16,
            (32, 16, 3),
            44,
        ),
        # Triple nesting: 128x128 -> 64x64 -> 32x32 -> 16x16
        (
            (128, 128, 3),
            [
                {"type": "tile", "configuration": {"tile_size": 64, "stride": 64}},
                {"type": "tile", "configuration": {"tile_size": 32, "stride": 32}},
                {"type": "tile", "configuration": {"tile_size": 16, "stride": 16}},
            ],
            64,
            (16, 16, 3),
            45,
        ),
    ],
)
def test_nested_tiling_lossless(pipeline, shape, steps, expected_final_count, expected_final_shape, seed):
    prep, recon = pipeline

    # Use the seed to ensure different random images
    rng = np.random.default_rng(seed)
    input_image = rng.integers(0, 256, shape, dtype=np.uint8)

    final_images, ops = prep.preprocess(input_image, steps)

    # Ensure the number of tiles is right
    assert len(final_images) == expected_final_count

    # Ensure the shapes of final images are right
    for img in final_images:
        assert img.shape == expected_final_shape

    restored_image = recon.reconstruct(final_images, ops)

    assert isinstance(restored_image, np.ndarray)
    assert restored_image.shape == input_image.shape
    assert np.allclose(restored_image, input_image)
