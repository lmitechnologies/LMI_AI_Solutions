import cv2
import numpy as np

from lmi_utils.label_utils.resize_with_csv import resize_imgs_with_csv


def _write(tmp_path, image):
    path_in, path_out = tmp_path / "in", tmp_path / "out"
    path_in.mkdir()
    path_out.mkdir()
    cv2.imwrite(str(path_in / "a.png"), image)
    path_csv = tmp_path / "labels.csv"
    path_csv.write_text("a.png;defect;1.0;rect;upper left;20;10\na.png;defect;1.0;rect;lower right;60;40\n")
    return path_in, path_csv, path_out


def test_image_and_labels_agree_when_height_already_matches(tmp_path):
    path_in, path_csv, path_out = _write(tmp_path, np.zeros((64, 128, 3), np.uint8))

    shapes = resize_imgs_with_csv(str(path_in), str(path_csv), [64, 64], str(path_out), False, False)

    assert cv2.imread(str(path_out / "a_resized_64x64.png")).shape[:2] == (64, 64)
    rect = shapes["a.png"][0]
    assert rect.up_left == [10, 10] and rect.bottom_right == [30, 40]


def test_resizes_with_the_bilinear_mode_inference_uses(tmp_path):
    image = np.random.default_rng(0).integers(0, 256, (96, 128, 3), dtype=np.uint8)
    path_in, path_csv, path_out = _write(tmp_path, image)

    resize_imgs_with_csv(str(path_in), str(path_csv), [40, 30], str(path_out), False, False)

    expected = cv2.resize(image, (40, 30), interpolation=cv2.INTER_LINEAR)
    assert np.array_equal(cv2.imread(str(path_out / "a_resized_40x30.png")), expected)
