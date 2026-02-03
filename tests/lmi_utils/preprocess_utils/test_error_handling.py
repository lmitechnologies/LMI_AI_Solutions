import numpy as np
import pytest
import torch
from preprocess_utils.preprocessor import Preprocessor
from preprocess_utils.reconstructor import Reconstructor


@pytest.fixture
def prep():
    """Returns a fresh Preprocessor instance for each test."""
    return Preprocessor()


@pytest.fixture
def recon():
    """Returns a fresh Reconstructor instance for each test."""
    return Reconstructor()


# ==========================================
# Group 1: Input Validation
# Tests related to data types and dimensions
# ==========================================


@pytest.mark.parametrize("invalid_input", [123, "string", None, 5.6, [None]])
def test_preprocessor_invalid_inputs(prep, invalid_input):
    """Test that both Preprocessor reject invalid image types."""
    with pytest.raises(TypeError, match="Images must be torch.Tensors or np.ndarrays"):
        prep.preprocess([invalid_input], [])


def test_preprocessor_one_dim_image(prep):
    """Test that preprocessor rejects one-dimensional images."""
    image = np.zeros((10,), dtype=np.uint8)
    tiler_meta = {"tile_size": [8, 8], "stride": [8, 8]}
    ops = [{"type": "tile", "configuration": tiler_meta}]

    with pytest.raises(ValueError, match="Input image must have 2 or 3 dimensions"):
        prep.preprocess([image], ops)


@pytest.mark.parametrize(
    "bad_input, error_type, match_msg",
    [
        ([], ValueError, "No input images provided"),
        (["not_an_image"], TypeError, "Images must be torch.Tensors"),
        ([torch.zeros(10, 10, 3), np.zeros((10, 10, 3))], TypeError, "All images must be the same type"),
        (np.zeros((10, 10, 3)), TypeError, "Images must be a list"),
    ],
)
def test_reconstructor_invalid_inputs(recon, bad_input, error_type, match_msg):
    """Test that reconstructor validates input structure and consistency."""
    with pytest.raises(error_type, match=match_msg):
        recon.reconstruct(bad_input, [])


# ==========================================
# Group 2: Step Configuration
# Tests related to step dict structure
# ==========================================


@pytest.mark.parametrize("invalid_step", [{"configuration": {}}, {"type": "resize"}])
def test_invalid_step_keys(prep, recon, invalid_step):
    """Test that both classes validate step structure (must have type and configuration)."""
    im = np.zeros((10, 10, 3), dtype=np.uint8)

    with pytest.raises(TypeError, match="Steps must be a list"):
        prep.preprocess(im, "not_a_list")
    with pytest.raises(TypeError, match="Steps must be a list"):
        recon.reconstruct([im], "not_a_list")

    with pytest.raises(TypeError, match="All steps must be dictionaries"):
        prep.preprocess(im, ["not_a_dict"])
    with pytest.raises(TypeError, match="All steps must be dictionaries"):
        recon.reconstruct([im], ["not_a_dict"])

    with pytest.raises(ValueError, match="Each step must contain keys"):
        prep.preprocess(im, [invalid_step])
    with pytest.raises(ValueError, match="Each step must contain keys"):
        recon.reconstruct([im], [invalid_step])


# ==========================================
# Group 3: Handler Registry
# Tests related to registering/retrieving handlers
# ==========================================


def test_unregistered_handler(prep, recon):
    """Test that classes reject operations with unknown handlers."""
    image = np.zeros((10, 10, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="Handler for 'unknown' is not registered"):
        prep.preprocess(image, [{"type": "unknown", "configuration": {}}])

    with pytest.raises(ValueError, match="Undo handler for 'unknown' is not registered"):
        recon.reconstruct([image], [{"type": "unknown", "configuration": {}}])


def test_register_non_callable(prep, recon):
    """Test that only callable functions can be registered."""
    with pytest.raises(TypeError, match="must be a callable function"):
        prep.register_handler("test", "not_callable")

    with pytest.raises(TypeError, match="Undo handler for 'test' must be callable"):
        recon.register_undo_handler("test", "not_callable")


# ==========================================
# Group 4: Handler Compliance
# Tests ensuring custom handlers return valid data
# ==========================================


def test_preprocessor_handler_returns(prep):
    """Test that preprocessor handlers return (list of tensors, dict)."""
    image = np.zeros((10, 10, 3), dtype=np.uint8)

    # 1. Returns single image instead of list
    def returns_non_list(images, config):
        return images[0], {}

    prep.register_handler("bad1", returns_non_list)
    with pytest.raises(TypeError, match="Handler 'bad1' must return a list of images"):
        prep.preprocess(image, [{"type": "bad1", "configuration": {}}])

    # 2. Returns non-dict metadata
    def returns_non_dict_metadata(images, config):
        return images, "not_a_dict"

    prep.register_handler("bad2", returns_non_dict_metadata)
    with pytest.raises(TypeError, match="Handler 'bad2' must return metadata as dict"):
        prep.preprocess(image, [{"type": "bad2", "configuration": {}}])

    # 3. Returns numpy arrays instead of tensors
    def returns_numpy(images, config):
        return [np.zeros((10, 10, 3))], {}

    prep.register_handler("bad3", returns_numpy)
    with pytest.raises(TypeError, match="Handler 'bad3' returned non-tensor images"):
        prep.preprocess(image, [{"type": "bad3", "configuration": {}}])


def test_reconstructor_handler_returns(recon):
    """Test that reconstructor undo handlers return list of tensors."""
    image = np.zeros((10, 10, 3), dtype=np.uint8)

    # 1. Returns single image instead of list
    def returns_non_list(images, metadata):
        return images[0]

    recon.register_undo_handler("bad1", returns_non_list)
    with pytest.raises(TypeError, match="Undo handler 'bad1' must return a list of images"):
        recon.reconstruct([image], [{"type": "bad1", "configuration": {}}])

    # 2. Returns numpy arrays instead of tensors
    def returns_numpy(images, metadata):
        return [np.zeros((10, 10, 3))]

    recon.register_undo_handler("bad2", returns_numpy)
    tensor_image = torch.zeros((10, 10, 3))
    with pytest.raises(TypeError, match="Undo handler 'bad2' returned non-tensor images"):
        recon.reconstruct([tensor_image], [{"type": "bad2", "configuration": {}}])


# ==========================================
# Group 5: Reconstruction Integrity
# Tests for logic specific to rebuilding images
# ==========================================


def test_tile_count_integrity(recon):
    """Test RuntimeError when tile count doesn't match images provided."""
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
