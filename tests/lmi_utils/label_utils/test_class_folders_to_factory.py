import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from lmi_utils.label_utils.class_folders_to_factory import convert_class_folders_to_factory, item_key


def write_images(root, paths, h=4, w=6):
    for path in paths:
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(root / path), np.zeros((h, w, 3), dtype=np.uint8))


def label_files(out):
    return {str(p.relative_to(out)): json.loads(p.read_text()) for p in out.rglob("*.label.json")}


def label_dictionary(out):
    return json.loads((out / ".meta.json").read_text())["labelDictionary"]


@pytest.mark.parametrize(
    "path, key",
    [("a/b.png", "a/b"), ("a/b.gadget2d.png", "a/b"), ("b.png", "b")],
    ids=["one extension", "compound extension", "top level"],
)
def test_item_key_strips_every_extension(path, key):
    assert item_key(Path(path)) == key


def test_each_image_is_labeled_with_its_class_folder(tmp_path):
    write_images(tmp_path / "in", ["ok/1.png", "ok/session/2.png", "bad/3.jpg"])
    counts = convert_class_folders_to_factory(tmp_path / "in", tmp_path / "out")

    assert counts == {"bad": 1, "ok": 2}
    files = label_files(tmp_path / "out")
    assert {path: file["annotations"][0]["label_id"] for path, file in files.items()} == {
        "ok/1.label.json": "ok",
        "ok/session/2.label.json": "ok",
        "bad/3.label.json": "bad",
    }
    assert files["ok/1.label.json"] == {"width": 6, "height": 4, "annotations": [{"id": "1", "type": "Classification", "label_id": "ok"}]}
    assert (tmp_path / "out" / "ok" / "session" / "2.png").is_file()
    assert label_dictionary(tmp_path / "out") == {
        "bad": {"id": "bad", "annotationType": "Classification", "order": 0},
        "ok": {"id": "ok", "annotationType": "Classification", "order": 1},
    }


def test_split_folders_hold_the_class_folders(tmp_path):
    write_images(tmp_path / "in", ["train/ok/1.png", "train/bad/2.png", "val/ok/3.png"])
    assert convert_class_folders_to_factory(tmp_path / "in", tmp_path / "out") == {"bad": 1, "ok": 2}
    assert set(label_files(tmp_path / "out")) == {"train/ok/1.label.json", "train/bad/2.label.json", "val/ok/3.label.json"}


def test_label_file_pairs_with_a_compound_extension_image(tmp_path):
    write_images(tmp_path / "in", ["ok/1.gadget2d.png"])
    convert_class_folders_to_factory(tmp_path / "in", tmp_path / "out")
    assert set(label_files(tmp_path / "out")) == {"ok/1.label.json"}


def test_per_class_samples_each_class(tmp_path):
    write_images(tmp_path / "in", [f"ok/{i}.png" for i in range(5)] + ["bad/0.png"])
    assert convert_class_folders_to_factory(tmp_path / "in", tmp_path / "out", per_class=2) == {"bad": 1, "ok": 2}


def test_link_shares_the_image_file(tmp_path):
    write_images(tmp_path / "in", ["ok/1.png"])
    convert_class_folders_to_factory(tmp_path / "in", tmp_path / "out", link=True)
    assert (tmp_path / "out" / "ok" / "1.png").samefile(tmp_path / "in" / "ok" / "1.png")


def test_hidden_folders_and_other_files_are_ignored(tmp_path):
    write_images(tmp_path / "in", ["ok/1.png", "ok/.cache/2.png", ".trash/3.png"])
    (tmp_path / "in" / "ok" / "notes.json").write_text("{}")
    assert convert_class_folders_to_factory(tmp_path / "in", tmp_path / "out") == {"ok": 1}


def test_a_class_without_images_is_left_out(tmp_path):
    write_images(tmp_path / "in", ["ok/1.png"])
    (tmp_path / "in" / "empty").mkdir()
    convert_class_folders_to_factory(tmp_path / "in", tmp_path / "out")
    assert list(label_dictionary(tmp_path / "out")) == ["ok"]


@pytest.mark.parametrize(
    "paths",
    [
        ["ok/1.png", "ok/1.jpg"],
        ["ok/1.png", "loose.png"],
        ["ok/1.png", "ok/2.tif"],
        ["ok/1.png", " spaced /2.png"],
    ],
    ids=["two images one item", "image outside a class", "image factory skips", "class name factory refuses"],
)
def test_problems_are_refused_before_writing(tmp_path, paths):
    write_images(tmp_path / "in", paths)
    with pytest.raises(ValueError):
        convert_class_folders_to_factory(tmp_path / "in", tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_a_non_empty_output_is_refused(tmp_path):
    write_images(tmp_path / "in", ["ok/1.png"])
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "stale.png").write_bytes(b"")
    with pytest.raises(ValueError):
        convert_class_folders_to_factory(tmp_path / "in", tmp_path / "out")
