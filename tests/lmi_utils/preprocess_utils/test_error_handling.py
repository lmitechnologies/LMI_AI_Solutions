import numpy as np
import pytest
import torch
from preprocess_utils.preprocessor import Preprocessor
from preprocess_utils.reconstructor import Reconstructor

# --- Preprocessor Error Handling Tests ---


def test_preprocessor_null_image():
    prep = Preprocessor()
    with pytest.raises(ValueError, match="Input image cannot be None"):
        prep.preprocess(None, [])


def test_preprocessor_invalid_step_keys():
    prep = Preprocessor()
    image = np.zeros((10, 10, 3), dtype=np.uint8)

    # Missing 'type'
    with pytest.raises(ValueError, match="Each processing step must contain keys"):
        prep.preprocess(image, [{"configuration": {}}])

    # Missing 'configuration'
    with pytest.raises(ValueError, match="Each processing step must contain keys"):
        prep.preprocess(image, [{"type": "resize"}])


def test_preprocessor_unregistered_handler():
    prep = Preprocessor()
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="Handler for 'unknown' is not registered"):
        prep.preprocess(image, [{"type": "unknown", "configuration": {}}])


def test_preprocessor_register_non_callable():
    prep = Preprocessor()
    with pytest.raises(TypeError, match="must be a callable function"):
        prep.register_handler("test", "not_callable")


# --- Reconstructor Error Handling Tests ---


def test_reconstructor_empty_images():
    recon = Reconstructor()
    with pytest.raises(ValueError, match="No input images provided"):
        recon.reconstruct([], [])


def test_reconstructor_invalid_image_type():
    recon = Reconstructor()
    with pytest.raises(TypeError, match="Images must be torch.Tensors or np.ndarrays"):
        recon.reconstruct(["not_an_image"], [])
    with pytest.raises(TypeError, match="All images must be the same type"):
        recon.reconstruct([torch.zeros((10, 10, 3)), np.zeros((10, 10, 3))], [])


def test_reconstructor_invalid_ops_keys():
    recon = Reconstructor()
    image = np.zeros((10, 10, 3), dtype=np.uint8)

    # Missing 'op'
    with pytest.raises(ValueError, match="Each operation step must contain keys"):
        recon.reconstruct([image], [{"metadata": {}}])

    # Missing 'metadata'
    with pytest.raises(ValueError, match="Each operation step must contain keys"):
        recon.reconstruct([image], [{"op": "resize"}])


def test_reconstructor_unsupported_undo():
    recon = Reconstructor()
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="No undo handler for unknown"):
        recon.reconstruct([image], [{"op": "unknown", "metadata": {}}])


def test_reconstructor_tile_integrity_failure():
    """
    Test RuntimeError when tile count doesn't match images provided.
    """
    recon = Reconstructor()
    image = torch.zeros((16, 16, 3))

    # Create metadata dict instead of mock object
    tiler_meta = {
        "n_tiles": [2, 2],  # Expects 4 tiles
    }

    meta = {"tiler_metadata": [tiler_meta]}
    ops = [{"op": "tile", "metadata": meta}]

    # Provide only 1 image instead of 4
    with pytest.raises(RuntimeError, match="Expected 4 tiles, found 1"):
        recon.reconstruct([image], ops)
