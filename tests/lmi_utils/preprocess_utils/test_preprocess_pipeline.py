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
        assert torch.allclose(restored_image, input_image)
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
    "input_shape, tile_configs, expected_count, final_shape",
    [
        ((77, 64, 3), [(32, 32), (16, 16)], 24, (16, 16, 3)),
        ((100, 100, 3), [(50, 25), (25, 25)], 36, (25, 25, 3)),
        ((128, 64, 3), [([64, 32], [64, 32]), ([32, 16], [32, 16])], 16, (32, 16, 3)),
        ((128, 128, 3), [(64, 64), (32, 32), (16, 16)], 64, (16, 16, 3)),
    ],
    ids=["simple", "overlap", "non_square", "triple_nesting"],
)
def test_nested_tiling_lossless(pipeline, input_shape, tile_configs, expected_count, final_shape):
    prep, recon = pipeline
    steps = [{"type": "tile", "configuration": {"tile_size": ts, "stride": st}} for ts, st in tile_configs]

    input_image = np.random.randint(0, 256, input_shape, dtype=np.uint8)
    final_images, ops = prep.preprocess(input_image, steps)

    # Ensure the number of tiles is right
    assert len(final_images) == expected_count

    # Ensure the shapes of final images are right
    for img in final_images:
        assert img.shape == final_shape

    restored_image = recon.reconstruct(final_images, ops)

    assert isinstance(restored_image, np.ndarray)
    assert restored_image.shape == input_image.shape
    assert np.allclose(restored_image, input_image)


@pytest.mark.parametrize("input_type", ["numpy", "torch"])
def test_incremental_reconstruction(pipeline, input_type):
    """Test that reconstruction works correctly at each stage of a multi-step pipeline."""

    def check_integrity(original, restored):
        is_list = isinstance(original, list)
        if not is_list:
            original = [original]
        is_list_restored = isinstance(restored, list)
        if not is_list_restored:
            restored = [restored]
        assert len(original) == len(restored)
        for o, r in zip(original, restored):
            if isinstance(o, torch.Tensor):
                assert torch.allclose(r, o)
            else:
                assert np.allclose(r, o)

    prep, recon = pipeline

    if input_type == "torch":
        input_images = [
            torch.randint(0, 256, (128, 128, 3), dtype=torch.uint8),
            torch.randint(0, 256, (121, 130, 3), dtype=torch.uint8),
            torch.randint(0, 256, (150, 140, 3), dtype=torch.uint8),
        ]
    else:
        input_images = [
            np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8),
            np.random.randint(0, 256, (121, 130, 3), dtype=np.uint8),
            np.random.randint(0, 256, (150, 140, 3), dtype=np.uint8),
        ]

    steps = [
        {"type": "resize", "configuration": {"width": 64, "height": 64}},
        {"type": "tile", "configuration": {"tile_size": 32, "stride": 32}},
        {"type": "tile", "configuration": {"tile_size": 16, "stride": 8}},
        {"type": "tile", "configuration": {"tile_size": 8, "stride": 4}},
    ]

    intermediate_states = []
    for i in range(len(steps)):
        images, ops = prep.preprocess(input_images, steps[: i + 1])
        intermediate_states.append((images, ops))

    # verify reconstruction at each stage
    for i in range(len(intermediate_states) - 1, 0, -1):  # skip resize because lossy
        inputs = intermediate_states[i - 1][0]
        images, ops = intermediate_states[i]
        restored = recon.reconstruct(images, ops[-1:])
        check_integrity(inputs, restored)


def test_tiling_signle_channel_image(pipeline):
    prep, recon = pipeline

    input_images = [
        torch.randint(0, 256, (100, 100), dtype=torch.uint8),  # Single channel image
        torch.randint(0, 256, (120, 130, 1), dtype=torch.uint8),  # Single channel with channel dim
        torch.randint(0, 256, (80, 90, 3), dtype=torch.uint8),  # Regular 3-channel image
    ]

    steps = [{"type": "tile", "configuration": {"tile_size": 50, "stride": 50}}]

    final_images, ops = prep.preprocess(input_images, steps)

    expected_tile_counts = [4, 9, 4]
    assert len(final_images) == sum(expected_tile_counts)
    for i in range(len(input_images)):
        expected_shape = (50, 50) if input_images[i].dim() == 2 else (50, 50, input_images[i].shape[2])
        for j in range(expected_tile_counts[i]):
            idx = sum(expected_tile_counts[:i]) + j
            assert final_images[idx].shape == expected_shape

    restored_images = recon.reconstruct(final_images, ops)

    for restored_image, input_image in zip(restored_images, input_images):
        assert isinstance(restored_image, torch.Tensor)
        assert restored_image.shape == input_image.shape
        assert torch.allclose(restored_image, input_image)
