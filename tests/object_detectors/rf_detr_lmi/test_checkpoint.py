"""The rfdetr entry point checkpoint.py calls, pinned so a signature change fails here rather than in the field."""

import argparse
import inspect
import os

import pytest
import torch
from rfdetr import RFDETR, RFDETRSegSmall

from object_detectors.rf_detr_lmi.checkpoint import _trust_kwarg, load_from_checkpoint

PTH_FILE = "tests/assets/models/od/rf_detr/rf-detr-seg-small.pth"
RESOLUTION = 384


def test_from_checkpoint_is_a_classmethod_on_the_base_class():
    """We call it off RFDETR itself and let it pick the subclass; an instance method would defeat that."""
    assert isinstance(inspect.getattr_static(RFDETR, "from_checkpoint"), classmethod)


def test_from_checkpoint_signature_matches_how_we_call_it():
    """checkpoint.py passes the path positionally and forwards **kwargs."""
    params = list(inspect.signature(RFDETR.from_checkpoint).parameters.values())

    assert params[0].kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    assert any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params), "no **kwargs left to forward device/resolution through"


def test_trust_checkpoint_is_passed_only_when_accepted():
    """rfdetr < 1.9 has no trust_checkpoint and feeds unknown kwargs to the model config, which rejects them."""
    params = inspect.signature(RFDETR.from_checkpoint).parameters
    passed = _trust_kwarg(RFDETR.from_checkpoint)

    assert passed == ({"trust_checkpoint": True} if "trust_checkpoint" in params else {})
    if passed:
        assert params["trust_checkpoint"].kind is not inspect.Parameter.POSITIONAL_ONLY


def test_returns_the_variant_the_checkpoint_records():
    model = load_from_checkpoint(PTH_FILE, ["small"], device="cpu")
    assert type(model) is RFDETRSegSmall
    assert model.model_config.resolution == RESOLUTION


def test_pretrain_weights_is_set_by_rfdetr():
    """checkpoint.py deliberately does not pass pretrain_weights; from_checkpoint must keep setting it itself."""
    model = load_from_checkpoint(PTH_FILE, ["small"], device="cpu")
    assert os.path.realpath(model.model_config.pretrain_weights) == os.path.realpath(PTH_FILE)


def test_num_classes_comes_off_the_checkpoint_head():
    """checkpoint.py omits num_classes so the head is sized by the weights; a rfdetr default would silently mismatch."""
    model = load_from_checkpoint(PTH_FILE, ["small"], device="cpu")
    assert model.model_config.num_classes == 90  # COCO, as stored in this checkpoint's class_embed


def test_resolution_kwarg_reaches_the_constructor():
    """Callers override the checkpoint's resolution through **kwargs; dropping it would export at the wrong size."""
    model = load_from_checkpoint(PTH_FILE, ["small"], device="cpu", resolution=432)
    assert model.model_config.resolution == 432


def test_namespace_args_still_load(tmp_path):
    """Checkpoints from rfdetr's pre-Lightning loop pickle args as an argparse.Namespace."""
    path = str(tmp_path / "namespace_args.pth")
    ckpt = torch.load(PTH_FILE, map_location="cpu", weights_only=False)
    ckpt["args"] = argparse.Namespace(**ckpt["args"])
    torch.save(ckpt, path)

    assert type(load_from_checkpoint(path, ["small"], device="cpu")) is RFDETRSegSmall


def test_missing_variant_raises_an_actionable_error(tmp_path):
    """rfdetr's own KeyError names nothing the caller can act on."""
    path = str(tmp_path / "no_variant.pth")
    torch.save({"model": {}}, path)
    with pytest.raises(ValueError, match="Specify model_type explicitly, one of: nano, small"):
        load_from_checkpoint(path, ["nano", "small"], device="cpu")
