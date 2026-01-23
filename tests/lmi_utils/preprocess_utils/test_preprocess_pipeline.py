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
    else:
        assert restored_image.shape == (128, 128, 3)
