import cv2
import numpy as np

from lmi_utils.dataset_utils.representations import Dataset, FileAnnotations
from lmi_utils.label_utils.json_to_factory import convert_json_to_factory


def test_annotation_files_import_beside_their_images(tmp_path):
    images = tmp_path / "images"
    for path in ("item/image.png", "flat/a.png"):
        (images / path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(images / path), np.zeros((4, 4, 3), dtype=np.uint8))
    files = [FileAnnotations(id=str(i), path=p, height=4, width=4) for i, p in enumerate(("item/image.png", "flat/a.png"))]

    convert_json_to_factory(Dataset(labels=[], files=files), images, tmp_path / "out")

    # an image named image.* is a Factory item folder, whose annotation file must be annotation.*
    assert (tmp_path / "out/item/annotation.label.json").is_file()
    assert (tmp_path / "out/flat/a.label.json").is_file()
