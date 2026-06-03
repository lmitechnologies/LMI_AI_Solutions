import json
import logging

import pytest
from pydantic import ValidationError

from lmi_utils.pipeline_base.core.schemas.schema_3 import (
    ADConfigs,
    ADModel,
    ModelCollectionV3,
    ODConfigs,
    ODModel,
)

logger = logging.getLogger(__name__)

KEYS = {"model_type", "model_path", "image_size", "algorithm", "package"}


@pytest.fixture
def schema3():
    with open("tests/lmi_utils/pipeline_base/schema3.json", "r") as f:
        return json.load(f)


def test_schema3(schema3):
    data = ModelCollectionV3.from_dict(schema3)
    models = data.get_metadata()

    assert isinstance(data, ModelCollectionV3)

    for _key, model in models.items():
        assert isinstance(model, dict)
        assert set(model.keys()) == KEYS
        assert len(model["model_path"]) > 0
        assert len(model["image_size"]) > 0
        assert len(model["algorithm"]) > 0
        assert len(model["package"]) > 0

    preprocessing = data.get_global_preprocessing()
    assert isinstance(preprocessing, dict)
    for _key, ops in preprocessing.items():
        assert isinstance(ops, list)
        for op in ops:
            assert isinstance(op, dict)
            assert "type" in op
            assert "configuration" in op
            assert op["type"] in {"resize", "tile"}
            if op["type"] == "tile":
                logger.info(f"Tile operation: {op['configuration']}")
                assert set(op["configuration"]) == {"tile_size", "stride"}
            if op["type"] == "resize":
                logger.info(f"Resize operation: {op['configuration']}")
                assert all(k in op["configuration"] for k in {"height", "width", "preserve_aspect"})


def test_discriminator_routes_ad(schema3):
    data = ModelCollectionV3.from_dict(schema3)
    ad = data.models["top_ad"]
    assert isinstance(ad, ADModel)
    assert isinstance(ad.configs, ADConfigs)
    assert ad.configs.min_threshold == 18.1
    assert ad.configs.max_threshold == 24.5


def test_discriminator_routes_od_and_alias(schema3):
    data = ModelCollectionV3.from_dict(schema3)
    od = data.models["top_od_defect"]
    assert isinstance(od, ODModel)
    assert isinstance(od.configs, ODConfigs)
    # "to-fail" alias must populate the to_fail field
    assert od.configs.to_fail["IL_Grooves"] is True
    assert od.configs.confidence["IL_Grooves"] == 0.5


def test_ad_model_missing_threshold_fails(schema3):
    schema3["top_ad"]["configs"].pop("min_threshold")
    with pytest.raises(ValidationError):
        ModelCollectionV3.from_dict(schema3)


def test_unknown_model_type_fails(schema3):
    schema3["top_ad"]["model_type"] = "Bogus"
    with pytest.raises(ValidationError):
        ModelCollectionV3.from_dict(schema3)


def test_none_entries_skipped(schema3):
    schema3["disabled_model"] = None
    mc = ModelCollectionV3.from_dict(schema3)
    assert "disabled_model" not in mc.models


def test_crop_to_label_step_ignored(schema3):
    # Legacy manifests may still carry a removed `crop-to-label` step; it is dropped
    # from the resolved chain rather than raising, for backward compatibility.
    steps = schema3["top_od_defect"]["details"]["preprocessing"]
    steps.insert(0, {"type": "crop-to-label", "configuration": {"label": "BOTTLE-BBOX"}})

    pre = ModelCollectionV3.from_dict(schema3).get_global_preprocessing()
    types = [op["type"] for op in pre["top_od_defect"]]
    assert "crop-to-label" not in types


def test_preprocessing_step_missing_id_stays_none(schema3):
    # Local manifests may omit `id`; it is not autofilled, and the resolved chain
    # omits the key entirely rather than emitting a null id.
    steps = schema3["top_od_defect"]["details"]["preprocessing"]
    for step in steps:
        step.pop("id", None)

    mc = ModelCollectionV3.from_dict(schema3)
    parsed = mc.models["top_od_defect"].details.preprocessing
    assert all(step.id is None for step in parsed)

    ops = mc.get_global_preprocessing()["top_od_defect"]
    assert all("id" not in op for op in ops)
