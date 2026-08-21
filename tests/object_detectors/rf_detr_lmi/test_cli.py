"""Which RF-DETR variant each CLI operation loads, and where it comes from."""

import os

import pytest
from rfdetr import RFDETRSegSmall, RFDETRSmall

from object_detectors.rf_detr_lmi import cli

PTH_FILE = "tests/assets/models/od/rf_detr/rf-detr-seg-small.pth"


@pytest.fixture
def loaded(monkeypatch):
    """Record which variant load_model resolves, without paying to build a real rfdetr model."""
    calls = {}

    def fake_from_checkpoint(model_path, supported, **kwargs):
        calls.update(source="checkpoint", model_path=model_path, supported=supported, kwargs=kwargs)
        return "model"

    def fake_get_model_class(task, model_type):
        def build(**kwargs):
            calls.update(source="model_type", task=task, model_type=model_type, kwargs=kwargs)
            return "model"

        return build

    def fake_convert_to_onnx(model, output_dir, **kwargs):
        path = os.path.join(output_dir, "rfdetr-seg-small.onnx")
        open(path, "w").close()
        return path

    monkeypatch.setattr(cli, "load_from_checkpoint", fake_from_checkpoint)
    monkeypatch.setattr(cli, "get_model_class", fake_get_model_class)
    monkeypatch.setattr(cli, "convert_to_onnx", fake_convert_to_onnx)
    return calls


def test_absent_model_type_is_not_defaulted():
    """No model_type must stay None, so convert/export can read the variant off the checkpoint."""
    configs = cli.parse_config(
        {"operation": "export", "task": "seg", "pretrain_weights": PTH_FILE, "export": {"output_dir": "/tmp", "resolution": 384}}
    )
    assert configs["model_configs"]["model_type"] is None


def test_convert_reads_variant_from_checkpoint(loaded):
    configs = cli.parse_config(
        {"operation": "convert", "task": "seg", "format": "onnx", "conversion": {"pretrain_weights": PTH_FILE, "resolution": 384}}
    )
    cli.load_model(configs)
    assert loaded["source"] == "checkpoint"
    assert loaded["model_path"] == PTH_FILE
    # pretrain_weights is left out: from_checkpoint sets it itself
    assert loaded["kwargs"] == {"resolution": 384}


@pytest.mark.parametrize(
    "config",
    [
        {"operation": "train", "task": "od", "training": {"output_dir": "/tmp"}},
        {"operation": "convert", "task": "seg", "format": "onnx", "conversion": {"pretrain_weights": PTH_FILE}},
        {"operation": "export", "task": "seg", "pretrain_weights": PTH_FILE, "export": {"output_dir": "/tmp"}},
    ],
    ids=["train", "convert", "export"],
)
def test_resolution_is_required(config):
    """The size is never read off the checkpoint, and an export bakes it in, so every operation must state it."""
    with pytest.raises(ValueError, match="resolution must be specified"):
        cli.parse_config(config)


def test_convert_model_type_overrides_checkpoint(loaded):
    configs = cli.parse_config(
        {
            "operation": "convert",
            "task": "seg",
            "format": "onnx",
            "model_type": "small",
            "conversion": {"pretrain_weights": PTH_FILE, "resolution": 384},
        }
    )
    cli.load_model(configs)
    assert loaded["source"] == "model_type"
    assert (loaded["task"], loaded["model_type"]) == ("seg", "small")
    assert loaded["kwargs"]["pretrain_weights"] == PTH_FILE


def test_convert_without_weights_uses_default_variant(loaded):
    """No checkpoint to read from — fall back to the default variant's own pretrained weights."""
    configs = cli.parse_config({"operation": "convert", "task": "od", "format": "onnx", "conversion": {"resolution": 384}})
    cli.load_model(configs)
    assert loaded["model_type"] == cli.DEFAULT_MODEL_TYPE


def test_export_reads_variant_from_checkpoint(loaded, tmp_path):
    configs = cli.parse_config(
        {
            "operation": "export",
            "task": "seg",
            "pretrain_weights": PTH_FILE,
            "export": {"output_dir": str(tmp_path), "resolution": 384},
        }
    )
    cli.handle_export(configs)
    assert loaded["source"] == "checkpoint"
    assert loaded["model_path"] == PTH_FILE
    assert loaded["kwargs"] == {"resolution": 384}
    assert os.path.exists(tmp_path / "model.onnx")


def test_export_model_type_overrides_checkpoint(loaded, tmp_path):
    configs = cli.parse_config(
        {
            "operation": "export",
            "task": "seg",
            "model_type": "small",
            "pretrain_weights": PTH_FILE,
            "export": {"output_dir": str(tmp_path), "resolution": 384},
        }
    )
    cli.handle_export(configs)
    assert loaded["source"] == "model_type"
    assert (loaded["task"], loaded["model_type"]) == ("seg", "small")


def test_train_still_defaults_the_variant(loaded):
    """Training may start from scratch, so it cannot read a variant and keeps the default."""
    configs = cli.parse_config({"operation": "train", "task": "od", "training": {"output_dir": "/tmp", "resolution": 384}})
    cli.load_model(configs)
    assert loaded["model_type"] == cli.DEFAULT_MODEL_TYPE


def test_supported_model_types_are_per_task():
    """seg registers sizes od does not; the error listing must not offer an unbuildable one."""
    assert "2xlarge" in cli.supported_model_types("seg")
    assert "2xlarge" not in cli.supported_model_types("od")


def test_checkpoint_variant_maps_onto_the_registry():
    """The name the checkpoint records must resolve to the same class as task: seg + model_type: small."""
    from object_detectors.rf_detr_lmi.checkpoint import load_from_checkpoint

    model = load_from_checkpoint(PTH_FILE, cli.supported_model_types("seg"), device="cpu")
    assert type(model) is RFDETRSegSmall
    assert cli.get_model_class("seg", "small") is RFDETRSegSmall
    assert cli.get_model_class("od", "small") is RFDETRSmall
