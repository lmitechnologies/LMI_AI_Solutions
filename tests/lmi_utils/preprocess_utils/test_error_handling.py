# Lightweight test-only Config/Meta/Operation triples for stub handlers.
from dataclasses import dataclass, field

import numpy as np
import pytest
import torch

from lmi_utils.preprocess_utils import steps
from lmi_utils.preprocess_utils.operation import Config, Meta, Operation
from lmi_utils.preprocess_utils.preprocessor import Preprocessor
from lmi_utils.preprocess_utils.reconstructor import Reconstructor


@dataclass
class _StubConfig(Config):
    pass


@dataclass
class _StubMeta(Meta):
    items: list = field(default_factory=list)


def _make_op(*, config_cls=None, meta_cls=None, forward=None, revert_images=None, revert_coords=None):
    if config_cls is None:
        config_cls = type("_TestCfg", (Config,), {"__annotations__": {}})
        config_cls = dataclass(config_cls)
    if meta_cls is None:
        meta_cls = type("_TestMeta", (Meta,), {"__annotations__": {}})
        meta_cls = dataclass(meta_cls)
    fwd = forward if forward is not None else (lambda self, images, config: (images, meta_cls()))
    attrs = {
        "config_cls": config_cls,
        "meta_cls": meta_cls,
        "forward": fwd,
    }
    if revert_images is not None:
        attrs["revert_images"] = revert_images
    if revert_coords is not None:
        attrs["revert_coords"] = revert_coords
    cls = type("_TestOp", (Operation,), attrs)
    return cls(), config_cls, meta_cls


@pytest.fixture
def prep():
    return Preprocessor()


@pytest.fixture
def recon():
    return Reconstructor()


@pytest.mark.parametrize("invalid_input", [123, "string", None, 5.6, [None]])
def test_preprocessor_invalid_inputs(prep, invalid_input):
    with pytest.raises(TypeError, match="Images must be torch.Tensors or np.ndarrays"):
        prep.preprocess([invalid_input], [])


def test_preprocessor_one_dim_image(prep):
    image = np.zeros((10,), dtype=np.uint8)
    with pytest.raises(ValueError, match="Expected 2D .HW. or 3D .HWC. image"):
        prep.preprocess([image], [steps.tile(tile_size=[8, 8], stride=[8, 8])])


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
    with pytest.raises(error_type, match=match_msg):
        recon.reconstruct_images(bad_input, [])


def test_configs_must_be_list(prep, recon):
    im = np.zeros((10, 10, 3), dtype=np.uint8)
    with pytest.raises(TypeError, match="configs must be a list"):
        prep.preprocess(im, "not_a_list")
    with pytest.raises(TypeError, match="history must be a list"):
        recon.reconstruct_images([im], "not_a_list")


def test_config_must_be_typed(prep):
    im = np.zeros((10, 10, 3), dtype=np.uint8)
    with pytest.raises(TypeError, match="configs\\[0\\] must be a Config"):
        prep.preprocess(im, ["not_a_config"])


def test_history_must_be_typed(recon):
    im = np.zeros((10, 10, 3), dtype=np.uint8)
    with pytest.raises(TypeError, match="history\\[0\\] must be a Meta instance"):
        recon.reconstruct_images([im], ["not_a_meta"])


def test_unregistered_config(prep):
    @dataclass
    class _UnregCfg(Config):
        pass

    image = np.zeros((10, 10, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="No Operation registered"):
        prep.preprocess(image, [_UnregCfg()])


def test_unregistered_meta(recon):
    @dataclass
    class _UnregMeta(Meta):
        pass

    image = np.zeros((10, 10, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="No Operation registered"):
        recon.reconstruct_images([image], [_UnregMeta()])


def test_register_non_operation(prep, recon):
    with pytest.raises(TypeError, match="Expected Operation"):
        prep.register("not_an_op")
    with pytest.raises(TypeError, match="Expected Operation"):
        recon.register("not_an_op")


def test_preprocessor_handler_returns(prep):
    image = np.zeros((10, 10, 3), dtype=np.uint8)

    op, cfg_cls, meta_cls = _make_op(forward=lambda self, images, config: (images[0], meta_cls()))
    prep.register(op)
    with pytest.raises(TypeError, match="must return a list of images"):
        prep.preprocess(image, [cfg_cls()])

    op2, cfg_cls2, meta_cls2 = _make_op(forward=lambda self, images, config: (images, "not_a_meta"))
    prep.register(op2)
    with pytest.raises(TypeError, match="expected Meta"):
        prep.preprocess(image, [cfg_cls2()])

    op3, cfg_cls3, meta_cls3 = _make_op(forward=lambda self, images, config: ([np.zeros((10, 10, 3))], meta_cls3()))
    prep.register(op3)
    with pytest.raises(TypeError, match="returned non-tensor images"):
        prep.preprocess(image, [cfg_cls3()])


def test_reconstructor_handler_returns(recon):
    image = np.zeros((10, 10, 3), dtype=np.uint8)

    op, _cfg, meta_cls = _make_op(revert_images=lambda self, images, meta: images[0])
    recon.register(op)
    with pytest.raises(TypeError, match="must return a list of images"):
        recon.reconstruct_images([image], [meta_cls()])

    op2, _cfg2, meta_cls2 = _make_op(revert_images=lambda self, images, meta: [np.zeros((10, 10, 3))])
    recon.register(op2)
    tensor_image = torch.zeros((10, 10, 3))
    with pytest.raises(TypeError, match="returned non-tensor images"):
        recon.reconstruct_images([tensor_image], [meta_cls2()])


def test_revert_coords_drops_field(recon):
    def drop_masks(self, results, meta):
        return [{k: v for k, v in r.items() if k != "masks"} for r in results]

    op, _cfg, meta_cls = _make_op(revert_coords=drop_masks)
    recon.register(op)

    results = {
        "boxes": [torch.tensor([[0.0, 0.0, 5.0, 5.0]])],
        "masks": [torch.ones((1, 10, 10))],
        "scores": [torch.tensor([0.9])],
        "classes": [np.array([0])],
    }
    with pytest.raises(KeyError, match="dropped non-empty coord field.*masks"):
        recon.reconstruct_coordinates(results, [meta_cls()])


def test_tile_count_integrity(recon):
    """Tile reconstruction must reject mismatched tile counts."""
    from lmi_utils.preprocess_utils.ops import TileMeta

    image = torch.zeros((16, 16, 3))
    meta = TileMeta(
        tile_sizes=[[8, 8]],
        strides=[[8, 8]],
        im_sizes=[[16, 16]],
        scale_sizes=[[16, 16]],
        n_tiles=[[2, 2]],
        batch_sizes=[1],
        num_channels=[3],
        scale_modes=["padding"],
        overlap_modes=["average"],
    )
    with pytest.raises(RuntimeError, match="Expected 4 tiles, found 1"):
        recon.reconstruct_images([image], [meta])
