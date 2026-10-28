import importlib
import subprocess
import sys

import pytest


@pytest.mark.parametrize("rel", ["csv_utils", "resize_with_csv", "shapes.mask"])
def test_old_path_is_the_moved_module(rel):
    old_name = f"lmi_utils.label_utils.{rel}"
    sys.modules.pop(old_name, None)
    with pytest.warns(FutureWarning, match="CSV label format is deprecated"):
        old = importlib.import_module(old_name)
    assert old is importlib.import_module(f"lmi_utils.label_utils.csv_labels.{rel}")


def test_old_shapes_package_shares_classes():
    from lmi_utils.label_utils.csv_labels import shapes

    with pytest.warns(FutureWarning):
        sys.modules.pop("lmi_utils.label_utils.shapes", None)
        old = importlib.import_module("lmi_utils.label_utils.shapes")
    assert old.Rect is shapes.Rect and old.Mask is shapes.Mask


def test_old_path_runs_as_script():
    out = subprocess.run([sys.executable, "-m", "lmi_utils.label_utils.csv_to_yolo", "--help"], capture_output=True, text=True, check=True)
    assert "usage:" in out.stdout
    assert "FutureWarning" in out.stderr
