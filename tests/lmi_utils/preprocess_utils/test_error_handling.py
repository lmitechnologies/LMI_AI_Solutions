import numpy as np
import pytest
import torch

from lmi_utils.preprocess_utils.operation import Operation
from lmi_utils.preprocess_utils.preprocessor import Preprocessor
from lmi_utils.preprocess_utils.reconstructor import Reconstructor


def _make_op(name, forward=None, revert_images=None, revert_coords=None):
    """Build a one-off Operation subclass with caller-provided behavior."""
    fwd = forward if forward is not None else (lambda images, config: (images, []))
    attrs = {
        "name": name,
        "forward": lambda self, images, config: fwd(images, config),
    }
    if revert_images is not None:
        attrs["revert_images"] = lambda self, images, metadata: revert_images(images, metadata)
    if revert_coords is not None:
        attrs["revert_coords"] = lambda self, results, metadata: revert_coords(results, metadata)
    return type("_TestOp", (Operation,), attrs)()


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
    ops = [{"type": "tile", "configuration": {"tile_size": [8, 8], "stride": [8, 8]}}]

    with pytest.raises(ValueError, match="Expected 2D .HW. or 3D .HWC. image"):
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
        recon.reconstruct_images(bad_input, [])


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
        recon.reconstruct_images([im], "not_a_list")

    with pytest.raises(TypeError, match="All steps must be dictionaries"):
        prep.preprocess(im, ["not_a_dict"])
    with pytest.raises(TypeError, match="All steps must be dictionaries"):
        recon.reconstruct_images([im], ["not_a_dict"])

    with pytest.raises(ValueError, match="Each step must contain keys"):
        prep.preprocess(im, [invalid_step])
    with pytest.raises(ValueError, match="Each step must contain keys"):
        recon.reconstruct_images([im], [invalid_step])


# ==========================================
# Group 3: Handler Registry
# Tests related to registering/retrieving handlers
# ==========================================


def test_unregistered_handler(prep, recon):
    """Test that classes reject operations with unknown handlers."""
    image = np.zeros((10, 10, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="Operation 'unknown' is not registered"):
        prep.preprocess(image, [{"type": "unknown", "configuration": {}}])

    with pytest.raises(ValueError, match="Revert image handler for 'unknown' is not registered"):
        recon.reconstruct_images([image], [{"type": "unknown", "metadata": []}])


def test_register_non_operation(prep, recon):
    """Test that only Operation instances can be registered."""
    with pytest.raises(TypeError, match="Expected Operation"):
        prep.register("not_an_op")
    with pytest.raises(TypeError, match="Expected Operation"):
        recon.register("not_an_op")


# ==========================================
# Group 4: Handler Compliance
# Tests ensuring custom handlers return valid data
# ==========================================


def test_preprocessor_handler_returns(prep):
    """Test that preprocessor ops return (list of tensors, dict)."""
    image = np.zeros((10, 10, 3), dtype=np.uint8)

    prep.register(_make_op("bad1", forward=lambda images, config: (images[0], [])))
    with pytest.raises(TypeError, match="Preprocess handler 'bad1' must return a list of images"):
        prep.preprocess(image, [{"type": "bad1", "configuration": {}}])

    prep.register(_make_op("bad2", forward=lambda images, config: (images, "not_a_list")))
    with pytest.raises(TypeError, match="Handler 'bad2' must return metadata as list"):
        prep.preprocess(image, [{"type": "bad2", "configuration": {}}])

    prep.register(_make_op("bad3", forward=lambda images, config: (images, {"wrapped": []})))
    with pytest.raises(TypeError, match="Handler 'bad3' must return metadata as list"):
        prep.preprocess(image, [{"type": "bad3", "configuration": {}}])

    prep.register(_make_op("bad4", forward=lambda images, config: ([np.zeros((10, 10, 3))], [])))
    with pytest.raises(TypeError, match="Preprocess handler 'bad4' returned non-tensor images"):
        prep.preprocess(image, [{"type": "bad4", "configuration": {}}])


def test_reconstructor_handler_returns(recon):
    """Test that reconstructor revert ops return list of tensors."""
    image = np.zeros((10, 10, 3), dtype=np.uint8)

    recon.register(_make_op("bad1", revert_images=lambda images, metadata: images[0]))
    with pytest.raises(TypeError, match="Revert image handler 'bad1' must return a list of images"):
        recon.reconstruct_images([image], [{"type": "bad1", "metadata": []}])

    recon.register(_make_op("bad2", revert_images=lambda images, metadata: [np.zeros((10, 10, 3))]))
    tensor_image = torch.zeros((10, 10, 3))
    with pytest.raises(TypeError, match="Revert image handler 'bad2' returned non-tensor images"):
        recon.reconstruct_images([tensor_image], [{"type": "bad2", "metadata": []}])


def test_revert_coords_drops_field(recon):
    """A coord handler that silently drops a populated field must raise."""

    def drop_masks(results, metadata):
        # Strip 'masks' from every per-image dict — simulates a buggy handler.
        return [{k: v for k, v in r.items() if k != "masks"} for r in results]

    recon.register(_make_op("dropper", revert_coords=drop_masks))

    results = {
        "boxes": [torch.tensor([[0.0, 0.0, 5.0, 5.0]])],
        "masks": [torch.ones((1, 10, 10))],
        "scores": [torch.tensor([0.9])],
        "classes": [np.array([0])],
    }
    steps = [{"type": "dropper", "metadata": [None]}]

    with pytest.raises(KeyError, match="dropped non-empty coord field.*masks"):
        recon.reconstruct_coordinates(results, steps)


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
    ops = [{"type": "tile", "metadata": [tiler_meta]}]

    # Provide only 1 image instead of 4
    with pytest.raises(RuntimeError, match="Expected 4 tiles, found 1"):
        recon.reconstruct_images([image], ops)
