import cv2
import numpy as np
import pytest
from PIL import ExifTags, Image, ImageOps
from PIL.TiffImagePlugin import ImageFileDirectory_v2

from lmi_utils.dataset_utils.file_utils import _EXIF_QUARTER_TURNS, update_file_dimensions
from lmi_utils.dataset_utils.representations import Dataset, FileAnnotations


def _write_image(path, h, w):
    cv2.imwrite(str(path / "image.png"), np.zeros((h, w, 3), dtype=np.uint8))


def _dataset(h, w):
    return Dataset(labels=[], files=[FileAnnotations("file", "image.png", h, w)])


ORIENTATION_TAG = ExifTags.Base.Orientation


def _write_tagged(path, name, h, w, orientation=None):
    """An image whose pixels are stored h x w, optionally tagged with an EXIF orientation."""
    image = Image.fromarray(np.zeros((h, w, 3), dtype=np.uint8))
    if orientation is None:
        image.save(str(path / name))
        return
    exif = image.getexif()
    exif[ORIENTATION_TAG] = orientation
    image.save(str(path / name), exif=exif)


def _write_tagged_tiff(path, name, h, w, orientation):
    """A tiff carrying a baseline orientation tag, which Pillow's reader applies to `.size` itself."""
    ifd = ImageFileDirectory_v2()
    ifd[ORIENTATION_TAG] = orientation
    Image.fromarray(np.zeros((h, w, 3), dtype=np.uint8)).save(str(path / name), tiffinfo=ifd)


def _dataset_for(name):
    return Dataset(labels=[], files=[FileAnnotations("file", name, None, None)])


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


@pytest.mark.parametrize("name", ["image.jpg", "image.png"])
@pytest.mark.parametrize("orientation", [5, 6, 7, 8])
def test_a_quarter_turn_orientation_reports_the_turned_size(tmp_path, name, orientation):
    """A camera can tag a quarter turn rather than rewrite the pixels, so the stored size is not the shown one."""
    _write_tagged(tmp_path, name, 10, 20, orientation)  # stored 20 wide x 10 tall, shown 10 wide x 20 tall

    dataset = update_file_dimensions(_dataset_for(name), str(tmp_path))

    assert (dataset.files[0].height, dataset.files[0].width) == (20, 10)


@pytest.mark.parametrize("name", ["image.jpg", "image.png"])
@pytest.mark.parametrize("orientation", [None, 1, 2, 3, 4])
def test_orientations_that_keep_the_axes_report_the_stored_size(tmp_path, name, orientation):
    """Only 5-8 transpose the axes; a mirror or a half turn leaves width and height where they were."""
    _write_tagged(tmp_path, name, 10, 20, orientation)

    dataset = update_file_dimensions(_dataset_for(name), str(tmp_path))

    assert (dataset.files[0].height, dataset.files[0].width) == (10, 20)


@pytest.mark.parametrize("name", ["image.jpg", "image.png"])
@pytest.mark.parametrize("orientation", [None, 1, 2, 3, 4, 5, 6, 7, 8])
def test_dimensions_match_the_pixels_opencv_decodes(tmp_path, name, orientation):
    """The contract these dimensions exist for: annotations are normalized against the decoded pixels.

    cv2 parses the tag with its own code and applies it on read, so this is what keeps the two libraries
    from disagreeing; reporting the stored size would swap the axes of every quarter-turned image and
    silently misplace its normalized coordinates.
    """
    _write_tagged(tmp_path, name, 10, 20, orientation)

    dataset = update_file_dimensions(_dataset_for(name), str(tmp_path))
    decoded_h, decoded_w = cv2.imread(str(tmp_path / name)).shape[:2]

    assert (dataset.files[0].height, dataset.files[0].width) == (decoded_h, decoded_w)


@pytest.mark.parametrize("orientation", [5, 6, 7, 8])
def test_a_tiff_orientation_is_not_applied_twice(tmp_path, orientation):
    """Pillow's tiff reader turns `.size` itself, so turning it again would put the image back.

    Written as stored 20 wide x 10 tall with a quarter turn tagged, which shows as 10 wide x 20 tall.
    """
    _write_tagged_tiff(tmp_path, "image.tiff", 10, 20, orientation)

    dataset = update_file_dimensions(_dataset_for("image.tiff"), str(tmp_path))

    assert (dataset.files[0].height, dataset.files[0].width) == (20, 10)


def test_the_turning_orientations_are_the_ones_pillow_itself_turns():
    """Which orientations transpose is the whole judgement in this fix, so read it off Pillow rather than
    trust a spec table copied by hand: exif_transpose changes the size for exactly these."""
    stored = Image.fromarray(np.zeros((10, 20, 3), dtype=np.uint8))

    turning = set()
    for orientation in range(1, 9):
        tagged = stored.copy()
        exif = tagged.getexif()
        exif[ORIENTATION_TAG] = orientation
        tagged.info["exif"] = exif.tobytes()
        if ImageOps.exif_transpose(tagged).size != stored.size:
            turning.add(orientation)

    assert turning == _EXIF_QUARTER_TURNS


def test_a_corrupt_exif_block_does_not_fail_a_readable_image(tmp_path):
    """A garbled tag must not cost the whole image; the pixels are still fine and the size is still stored."""
    _write_tagged(tmp_path, "image.jpg", 10, 20, 6)
    raw = (tmp_path / "image.jpg").read_bytes()
    marker = raw.find(b"Exif\x00\x00")
    (tmp_path / "image.jpg").write_bytes(raw[:marker] + b"Exif\x00\x00" + b"\xde\xad\xbe\xef" * 4 + raw[marker + 22 :])

    dataset = update_file_dimensions(_dataset_for("image.jpg"), str(tmp_path))

    assert (dataset.files[0].height, dataset.files[0].width) == (10, 20)
