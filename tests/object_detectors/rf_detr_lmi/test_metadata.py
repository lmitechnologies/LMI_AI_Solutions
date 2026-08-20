"""The rfdetr export payload: one owner for the key names and the fallbacks.

The exporter and the engine backends read these fields from opposite ends of a deployed file, so a
drift between them is silent — the model loads and postprocesses with the wrong num_select.
"""

import logging
from types import SimpleNamespace

import pytest

from object_detectors.rf_detr_lmi.metadata import DEFAULT_NUM_SELECT, RfdetrMetadata

NAMES = ["cat", "dog"]


def _fake_rfdetr_model(class_names, num_select):
    """The two attribute paths from_model reaches through on a real rfdetr model."""
    return SimpleNamespace(class_names=class_names, model=SimpleNamespace(postprocess=SimpleNamespace(num_select=num_select)))


def test_round_trip_through_the_payload():
    """What the exporter writes is what an engine backend reads back."""
    written = RfdetrMetadata.from_model(_fake_rfdetr_model(NAMES, 100)).as_payload()
    assert RfdetrMetadata.from_engine(written, "model.engine") == RfdetrMetadata(class_names=NAMES, num_select=100)


def test_from_model_copies_the_names():
    """The payload must not alias a list the model may go on to mutate."""
    names = list(NAMES)
    metadata = RfdetrMetadata.from_model(_fake_rfdetr_model(names, 100))
    names.append("bird")
    assert metadata.class_names == NAMES


def test_payload_leaves_out_what_it_has_nothing_to_say_about():
    assert RfdetrMetadata(num_select=100).as_payload() == {"num_select": 100}


def test_a_model_without_metadata_falls_back_loudly(caplog):
    """A wrong num_select silently changes how many detections come back, so it must not pass unremarked."""
    with caplog.at_level(logging.WARNING):
        metadata = RfdetrMetadata.from_engine({}, "model.engine")
    assert metadata == RfdetrMetadata(class_names=None, num_select=DEFAULT_NUM_SELECT)
    assert "num_select" in caplog.text and "model.engine" in caplog.text


def test_embedded_num_select_is_not_warned_about(caplog):
    with caplog.at_level(logging.WARNING):
        assert RfdetrMetadata.from_engine({"class_names": NAMES, "num_select": 100}, "model.engine").num_select == 100
    assert caplog.text == ""


@pytest.mark.parametrize("class_names", [None, []])
def test_absent_names_normalize_to_none(class_names):
    """One 'no names here' value, so the caller has a single case to raise on."""
    assert RfdetrMetadata.from_engine({"class_names": class_names, "num_select": 100}, "model.engine").class_names is None
