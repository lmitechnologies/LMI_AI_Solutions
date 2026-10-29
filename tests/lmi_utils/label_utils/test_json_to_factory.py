import cv2
import numpy as np
import pytest

from lmi_utils.dataset_utils.representations import Dataset, FileAnnotations
from lmi_utils.label_utils.json_to_factory import convert_json_to_factory


def convert(tmp_path, paths):
    images = tmp_path / "images"
    for path in paths:
        (images / path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(images / path), np.zeros((4, 4, 3), dtype=np.uint8))
    files = [FileAnnotations(id=str(i), path=p, height=4, width=4) for i, p in enumerate(paths)]
    convert_json_to_factory(Dataset(labels=[], files=files), images, tmp_path / "out")
    return sorted(str(p.relative_to(tmp_path / "out")) for p in (tmp_path / "out").rglob("*.label.json"))


@pytest.mark.parametrize(
    "paths, expected",
    [
        (["a/image.png", "b/image.png"], ["a/annotation.label.json", "b/annotation.label.json"]),
        # Factory groups a mixed dataset by path without extension, which pairs image.label.json with image.png
        (["a/image.png", "flat/b.png"], ["a/image.label.json", "flat/b.label.json"]),
    ],
    ids=["item folders", "mixed"],
)
def test_annotation_files_pair_with_their_images_on_import(tmp_path, paths, expected):
    assert convert(tmp_path, paths) == expected
