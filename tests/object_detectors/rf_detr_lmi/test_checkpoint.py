"""The rfdetr entry point checkpoint.py calls, pinned so a signature change fails here rather than in the field."""

import argparse
import inspect
import os
from types import SimpleNamespace

import pytest
import torch
from rfdetr import RFDETR, RFDETRSegSmall, variants

from object_detectors.rf_detr_lmi.checkpoint import (
    _pe_tracks_resolution,
    _resolution_from_weights,
    _trust_kwarg,
    load_from_checkpoint,
    resolution_from_checkpoint,
)

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


def test_resolution_from_checkpoint_reads_the_recorded_value():
    assert resolution_from_checkpoint(PTH_FILE) == RESOLUTION


def test_resolution_falls_back_to_the_weights():
    """rfdetr strips model_config out of checkpoint_best_total.pth, leaving the position grid as the only record."""
    checkpoint = torch.load(PTH_FILE, map_location="cpu", weights_only=False)
    del checkpoint["model_config"]
    assert _resolution_from_weights(checkpoint["model"]) == RESOLUTION


def test_weights_derivation_matches_a_live_model():
    """Pins the state-dict names and the grid-times-patch rule to this rfdetr; a rename here beats a silent wrong size."""
    model = load_from_checkpoint(PTH_FILE, ["small"], device="cpu")
    assert _resolution_from_weights(model.model.model.state_dict()) == model.model_config.resolution


def test_no_derivation_for_a_variant_that_pins_its_position_grid(monkeypatch, tmp_path):
    """Deriving the resolution is only valid where rfdetr itself sizes the grid from the resolution."""

    class PinnedConfig:
        model_fields = {
            "positional_encoding_size": SimpleNamespace(default=37),
            "patch_size": SimpleNamespace(default=14),
            "resolution": SimpleNamespace(default=560),
        }

    monkeypatch.setattr(variants, "RFDETRPinned", SimpleNamespace(_model_config_class=PinnedConfig), raising=False)
    assert _pe_tracks_resolution("RFDETRPinned") is False

    path = str(tmp_path / "pinned.pth")
    checkpoint = torch.load(PTH_FILE, map_location="cpu", weights_only=False)
    del checkpoint["model_config"]
    checkpoint["model_name"] = "RFDETRPinned"
    torch.save(checkpoint, path)
    assert resolution_from_checkpoint(path) is None


def test_every_supported_variant_sizes_its_grid_from_its_resolution():
    """The rule the weights fallback relies on; a new variant that breaks it must show up here."""
    assert all(_pe_tracks_resolution(f"RFDETR{name}") for name in ("Nano", "Small", "Medium", "Large", "SegSmall", "SegLarge"))


def test_resolution_from_checkpoint_is_none_when_unreadable(tmp_path):
    """Nothing to read the resolution from; the caller then leaves it to rfdetr."""
    path = str(tmp_path / "no_model_config.pth")
    torch.save({"model": {}}, path)
    assert resolution_from_checkpoint(path) is None


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
