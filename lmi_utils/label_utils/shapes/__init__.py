"""Deprecated path: moved to lmi_utils.label_utils.csv_labels.shapes."""

from lmi_utils.label_utils.csv_labels import warn_moved
from lmi_utils.label_utils.csv_labels.shapes import Brush, Keypoint, Mask, Rect, Shape  # noqa: F401

warn_moved("shapes")
