import pytest

from lmi_utils.dataset_utils.pose_identifiers import (
    POSE_ID_MAX_LENGTH,
    derive_pose_id,
    is_pose_id,
    is_reserved_pose_id,
    pose_id_from_label_alias,
    pose_label_alias,
)

# Derivation is a persisted contract shared with the GoFactory TypeScript and Python SDKs: a dataset converted
# today and one converted next year must reach the same ids, in whichever language did the converting. These
# expectations are therefore written out in full rather than computed. The same table exists in the SDKs; a
# change to any one copy has to be made to all three, and this test is what says so.
DERIVATION_VECTORS = [
    # A name already usable as an id is kept, which is what keeps ids readable wherever a person sees them
    ("person", "person"),
    ("left_eye", "left_eye"),
    ("Housing", "Housing"),
    ("left-flange", "left-flange"),
    ("point-0", "point-0"),
    # Anything else becomes a readable stem plus a digest of the whole name
    ("traffic light", "traffic_light_34d739963577"),
    ("Left Eye", "left_eye_4c96d10f1b2a"),
    # The digest is of the source name, so names that share a stem stay distinct concepts
    ("left eye", "left_eye_48a09a9e26d4"),
    # A stem keeps ASCII only, and is folded case-insensitively without consulting any locale
    ("café", "caf_850f7dc43910"),
    ("左目", "label_46efb21fc5d3"),
    ("🙂", "label_d06f1525f791"),
    # Whitespace is trimmed to ECMAScript's set, which includes the byte-order mark and excludes what Python's
    # own str.strip() would additionally take
    ("  person  ", "person"),
    ("﻿person﻿", "person"),
    # A name forging the alias prefix derives an id that no longer carries it
    ("__fsp_person_0123456789ab", "fsp_person_0123456789ab_21445e379b30"),
]


@pytest.mark.parametrize("source_name, expected", DERIVATION_VECTORS)
def test_derivation_matches_the_shared_contract(source_name, expected):
    assert derive_pose_id(source_name) == expected


@pytest.mark.parametrize("source_name, expected", DERIVATION_VECTORS)
def test_every_derived_id_is_usable_and_unreserved(source_name, expected):
    assert is_pose_id(expected)
    assert not is_reserved_pose_id(expected)


def test_a_long_name_still_derives_a_usable_id():
    # The stem is truncated to leave room for the digest, which is what carries identity, so two long names
    # sharing a prefix stay distinct.
    first = derive_pose_id("a very long keypoint name " * 20)
    second = derive_pose_id("a very long keypoint name " * 20 + "tip")

    assert len(first) <= POSE_ID_MAX_LENGTH
    assert is_pose_id(first)
    assert first != second


def test_an_alias_round_trips_to_its_id():
    assert pose_id_from_label_alias(pose_label_alias("left_eye")) == "left_eye"


def test_a_value_outside_the_namespace_is_left_alone():
    # A project Factory did not generate names its labels however it likes; those are observations to be checked
    # against the schema, not values to be rewritten.
    assert pose_id_from_label_alias("left_eye") == "left_eye"
