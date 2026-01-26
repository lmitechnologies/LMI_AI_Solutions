import numpy as np
import pytest
import torch
from preprocess_utils.preprocessor import Preprocessor
from preprocess_utils.reconstructor import Reconstructor

# --- Preprocessor Error Handling Tests ---


def test_preprocessor_null_image():
    prep = Preprocessor()
    with pytest.raises(ValueError, match="No input image provided for preprocessing"):
        prep.preprocess(None, [])


def test_preprocessor_invalid_image_type():
    """Test that preprocessor rejects invalid image types."""
    prep = Preprocessor()
    with pytest.raises(TypeError, match="Images must be torch.Tensors or np.ndarrays"):
        prep.preprocess("not_an_image", [])


def test_preprocessor_invalid_step_keys():
    prep = Preprocessor()
    image = np.zeros((10, 10, 3), dtype=np.uint8)

    # Missing 'type'
    with pytest.raises(ValueError, match="Each step must contain keys"):
        prep.preprocess(image, [{"configuration": {}}])

    # Missing 'configuration'
    with pytest.raises(ValueError, match="Each step must contain keys"):
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


def test_preprocessor_handler_returns_non_list():
    """Test that handlers must return a list of images."""
    prep = Preprocessor()

    def bad_handler(images, config):
        return images[0], {}  # Returns single image, not list

    prep.register_handler("bad", bad_handler)
    image = np.zeros((10, 10, 3), dtype=np.uint8)

    with pytest.raises(TypeError, match="Handler 'bad' must return a list of images"):
        prep.preprocess(image, [{"type": "bad", "configuration": {}}])


def test_preprocessor_handler_returns_non_dict_metadata():
    """Test that handlers must return metadata as dict."""
    prep = Preprocessor()

    def bad_handler(images, config):
        return images, "not_a_dict"  # Returns string instead of dict

    prep.register_handler("bad", bad_handler)
    image = np.zeros((10, 10, 3), dtype=np.uint8)

    with pytest.raises(TypeError, match="Handler 'bad' must return metadata as dict"):
        prep.preprocess(image, [{"type": "bad", "configuration": {}}])


def test_preprocessor_handler_returns_non_tensors():
    """Test that handlers must return torch tensors."""
    prep = Preprocessor()

    def bad_handler(images, config):
        return [np.zeros((10, 10, 3))], {}  # Returns numpy array

    prep.register_handler("bad", bad_handler)
    image = np.zeros((10, 10, 3), dtype=np.uint8)

    with pytest.raises(TypeError, match="Handler 'bad' returned non-tensor images"):
        prep.preprocess(image, [{"type": "bad", "configuration": {}}])


# --- Reconstructor Error Handling Tests ---


def test_reconstructor_empty_images():
    recon = Reconstructor()
    with pytest.raises(ValueError, match="No input images provided for reconstruction"):
        recon.reconstruct([], [])


def test_reconstructor_non_list_input():
    """Test that reconstructor requires a list of images."""
    recon = Reconstructor()
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    with pytest.raises(TypeError, match="Images must be a list"):
        recon.reconstruct(image, [])  # Passing single image instead of list


def test_reconstructor_invalid_image_type():
    recon = Reconstructor()
    with pytest.raises(TypeError, match="Images must be torch.Tensors or np.ndarrays"):
        recon.reconstruct(["not_an_image"], [])


def test_reconstructor_mixed_image_types():
    """Test that all images must be the same type."""
    recon = Reconstructor()
    with pytest.raises(TypeError, match="All images must be the same type"):
        recon.reconstruct([torch.zeros((10, 10, 3)), np.zeros((10, 10, 3))], [])


def test_reconstructor_invalid_step_keys():
    recon = Reconstructor()
    image = np.zeros((10, 10, 3), dtype=np.uint8)

    # Missing 'op'
    with pytest.raises(ValueError, match="Each step must contain keys"):
        recon.reconstruct([image], [{"metadata": {}}])

    # Missing 'metadata'
    with pytest.raises(ValueError, match="Each step must contain keys"):
        recon.reconstruct([image], [{"op": "resize"}])


def test_reconstructor_unregistered_undo_handler():
    recon = Reconstructor()
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="Undo handler for 'unknown' is not registered"):
        recon.reconstruct([image], [{"op": "unknown", "metadata": {}}])


def test_reconstructor_register_non_callable():
    """Test that undo handlers must be callable."""
    recon = Reconstructor()
    with pytest.raises(TypeError, match="Undo handler for 'test' must be callable"):
        recon.register_undo_handler("test", "not_callable")


def test_reconstructor_undo_handler_returns_non_list():
    """Test that undo handlers must return a list of images."""
    recon = Reconstructor()

    def bad_undo_handler(images, metadata):
        return images[0]  # Returns single image, not list

    recon.register_undo_handler("bad", bad_undo_handler)
    image = np.zeros((10, 10, 3), dtype=np.uint8)

    with pytest.raises(TypeError, match="Undo handler 'bad' must return a list of images"):
        recon.reconstruct([image], [{"op": "bad", "metadata": {}}])


def test_reconstructor_undo_handler_returns_non_tensors():
    """Test that undo handlers must return torch tensors."""
    recon = Reconstructor()

    def bad_undo_handler(images, metadata):
        return [np.zeros((10, 10, 3))]  # Returns numpy array

    recon.register_undo_handler("bad", bad_undo_handler)
    image = torch.zeros((10, 10, 3))

    with pytest.raises(TypeError, match="Undo handler 'bad' returned non-tensor images"):
        recon.reconstruct([image], [{"op": "bad", "metadata": {}}])


def test_reconstructor_multiple_images_after_reconstruction():
    """Test that reconstruction must result in exactly 1 image."""
    recon = Reconstructor()

    def bad_undo_handler(images, metadata):
        # Returns 2 images instead of combining to 1
        return [torch.zeros((10, 10, 3)), torch.zeros((10, 10, 3))]

    recon.register_undo_handler("bad", bad_undo_handler)
    image = np.zeros((10, 10, 3), dtype=np.uint8)

    with pytest.raises(RuntimeError, match="Reconstruction expected 1 image, got 2"):
        recon.reconstruct([image], [{"op": "bad", "metadata": {}}])


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
