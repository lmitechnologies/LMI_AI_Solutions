import cv2
import numpy as np
import pytest

from lmi_utils.dataset_utils.file_utils import update_file_dimensions
from lmi_utils.dataset_utils.representations import Dataset, FileAnnotations


def _write_image(path, h, w):
    cv2.imwrite(str(path / "image.png"), np.zeros((h, w, 3), dtype=np.uint8))


def _dataset(h, w):
    return Dataset(labels=[], files=[FileAnnotations("file", "image.png", h, w)])


@pytest.mark.parametrize("stated", [(None, None), (0, 0), (30, 40), (10, None)])
def test_dimensions_always_come_from_the_image(tmp_path, stated):
    # The image on disk governs, whatever the dataset states -- annotations are normalized against these.
    _write_image(tmp_path, 10, 20)

    dataset = update_file_dimensions(_dataset(*stated), str(tmp_path))

    assert (dataset.files[0].height, dataset.files[0].width) == (10, 20)


def test_missing_image_raises(tmp_path):
    with pytest.raises(Exception, match="File not found"):
        update_file_dimensions(_dataset(30, 40), str(tmp_path))


def test_unreadable_image_raises(tmp_path):
    (tmp_path / "image.png").write_text("not an image")

    with pytest.raises(Exception, match="cannot read image"):
        update_file_dimensions(_dataset(None, None), str(tmp_path))
