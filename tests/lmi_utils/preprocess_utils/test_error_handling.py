import numpy as np
import pytest
import torch
from preprocess_utils.preprocessor import Preprocessor
from preprocess_utils.reconstructor import Reconstructor


@pytest.mark.parametrize("invalid_input", [123, "string", None, 5.6, [None]])
def test_preprocessor_invalid_image_type(invalid_input):
    """Test that preprocessor rejects invalid image types."""
    prep = Preprocessor()
    with pytest.raises(TypeError, match="Images must be torch.Tensors or np.ndarrays"):
        prep.preprocess([invalid_input], [])


def test_one_dim_image_tiling():
    """Test that preprocessor rejects one-dimensional images."""
    prep = Preprocessor()
    image = np.zeros((10,), dtype=np.uint8)
    tiler_meta = {
        "tile_size": [8, 8],
        "stride": [8, 8],
    }

    ops = [{"type": "tile", "configuration": tiler_meta}]
    with pytest.raises(ValueError, match="Input image must have 2 or 3 dimensions"):
        prep.preprocess([image], ops)


@pytest.mark.parametrize(
    "invalid_step",
    [
        {"configuration": {}},
        {"type": "resize"},
    ],
)
def test_preprocessor_invalid_step_keys(invalid_step):
    """Test that preprocessor validates step structure."""
    prep = Preprocessor()
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="Each step must contain keys"):
        prep.preprocess(image, [invalid_step])


def test_preprocessor_unregistered_handler():
    """Test that preprocessor rejects unregistered handlers."""
    prep = Preprocessor()
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="Handler for 'unknown' is not registered"):
        prep.preprocess(image, [{"type": "unknown", "configuration": {}}])


def test_preprocessor_register_non_callable():
    """Test that only callable handlers can be registered."""
    prep = Preprocessor()
    with pytest.raises(TypeError, match="must be a callable function"):
        prep.register_handler("test", "not_callable")


def test_preprocessor_handler_invalid_returns():
    """Test that handlers must return (list of tensors, dict)."""
    prep = Preprocessor()
    image = np.zeros((10, 10, 3), dtype=np.uint8)

    # Handler returns single image instead of list
    def returns_non_list(images, config):
        return images[0], {}

    prep.register_handler("bad1", returns_non_list)
    with pytest.raises(TypeError, match="Handler 'bad1' must return a list of images"):
        prep.preprocess(image, [{"type": "bad1", "configuration": {}}])

    # Handler returns non-dict metadata
    def returns_non_dict_metadata(images, config):
        return images, "not_a_dict"

    prep.register_handler("bad2", returns_non_dict_metadata)
    with pytest.raises(TypeError, match="Handler 'bad2' must return metadata as dict"):
        prep.preprocess(image, [{"type": "bad2", "configuration": {}}])

    # Handler returns numpy arrays instead of tensors
    def returns_numpy(images, config):
        return [np.zeros((10, 10, 3))], {}

    prep.register_handler("bad3", returns_numpy)
    with pytest.raises(TypeError, match="Handler 'bad3' returned non-tensor images"):
        prep.preprocess(image, [{"type": "bad3", "configuration": {}}])


def test_reconstructor_invalid_inputs():
    """Test that reconstructor validates input images."""
    recon = Reconstructor()

    # Empty list
    with pytest.raises(ValueError, match="No input images provided for reconstruction"):
        recon.reconstruct([], [])

    # Invalid image type
    with pytest.raises(TypeError, match="Images must be torch.Tensors or np.ndarrays"):
        recon.reconstruct(["not_an_image"], [])

    # Mixed image types
    with pytest.raises(TypeError, match="All images must be the same type"):
        recon.reconstruct([torch.zeros((10, 10, 3)), np.zeros((10, 10, 3))], [])


@pytest.mark.parametrize(
    "invalid_step",
    [
        {"configuration": {}},
        {"type": "resize"},
    ],
)
def test_reconstructor_invalid_step_keys(invalid_step):
    """Test that reconstructor validates step structure."""
    recon = Reconstructor()
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="Each step must contain keys"):
        recon.reconstruct([image], [invalid_step])


def test_reconstructor_unregistered_undo_handler():
    """Test that reconstructor rejects unregistered undo handlers."""
    recon = Reconstructor()
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="Undo handler for 'unknown' is not registered"):
        recon.reconstruct([image], [{"type": "unknown", "configuration": {}}])


def test_reconstructor_register_non_callable():
    """Test that only callable undo handlers can be registered."""
    recon = Reconstructor()
    with pytest.raises(TypeError, match="Undo handler for 'test' must be callable"):
        recon.register_undo_handler("test", "not_callable")


def test_reconstructor_undo_handler_invalid_returns():
    """Test that undo handlers must return list of tensors."""
    recon = Reconstructor()

    # Undo handler returns single image instead of list
    def returns_non_list(images, metadata):
        return images[0]

    recon.register_undo_handler("bad1", returns_non_list)
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    with pytest.raises(TypeError, match="Undo handler 'bad1' must return a list of images"):
        recon.reconstruct([image], [{"type": "bad1", "configuration": {}}])

    # Undo handler returns numpy arrays instead of tensors
    def returns_numpy(images, metadata):
        return [np.zeros((10, 10, 3))]

    recon.register_undo_handler("bad2", returns_numpy)
    image = torch.zeros((10, 10, 3))
    with pytest.raises(TypeError, match="Undo handler 'bad2' returned non-tensor images"):
        recon.reconstruct([image], [{"type": "bad2", "configuration": {}}])


def test_reconstructor_tile_integrity_failure():
    """Test RuntimeError when tile count doesn't match images provided."""
    recon = Reconstructor()
    image = torch.zeros((16, 16, 3))

    tiler_meta = {
        "n_tiles": [2, 2],  # Expects 4 tiles
        "tile_size": [8, 8],
        "stride": [8, 8],
    }

    meta = {"tiler_metadata": [tiler_meta]}
    ops = [{"type": "tile", "configuration": meta}]

    # Provide only 1 image instead of 4
    with pytest.raises(RuntimeError, match="Expected 4 tiles, found 1"):
        recon.reconstruct([image], ops)
