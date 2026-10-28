"""Label Studio project conventions shared by the LST writers: control tag names, image URLs and the labeling interface."""

import logging
import xml.etree.ElementTree as ET
from typing import List

logger = logging.getLogger(__name__)

XML_NAME = "labeling_interface.xml"
# must match the control tag names in the Label Studio labeling interface; "label" keeps csv_to_lst projects working
CONTROL_NAMES = {
    "rectanglelabels": ("label", "RectangleLabels"),
    "polygonlabels": ("polygon", "PolygonLabels"),
    "brushlabels": ("brush", "BrushLabels"),
    "keypointlabels": ("keypoint", "KeyPointLabels"),
}
CONTROL_ATTRS = {"polygonlabels": {"strokeWidth": "3", "pointSize": "small", "opacity": "0.9"}}


def to_image_root(path: str) -> str:
    """Return the prefix joined with each image path; a local absolute path becomes a Label Studio local-files URL.

    The first folder of a local path is taken as LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT.
    """
    if path.startswith("gs://"):
        return path.rstrip("/") + "/"
    if not path.startswith("/"):
        raise ValueError("The local storage path must be absolute path starting with /")
    parts = [p for p in path.split("/") if p]
    if not parts:
        raise ValueError("The local storage path must include the document root folder")
    logger.info(f"Assume that LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT is /{parts[0]}")
    return "/data/local-files/?d=" + "".join(p + "/" for p in parts[1:])


def labeling_interface(tasks: List[dict]) -> str:
    """Build a labeling interface XML with one control tag per result type found in the tasks."""
    classes = {}
    for task in tasks:
        for key in ("annotations", "predictions"):
            for result in (r for item in task.get(key, []) for r in item["result"]):
                classes.setdefault(result["type"], set()).update(result["value"][result["type"]])

    root = ET.Element("View")
    ET.SubElement(root, "Image", name="image", value="$image", zoom="true")
    for result_type, (name, tag) in CONTROL_NAMES.items():
        if result_type in classes:
            control = ET.SubElement(root, tag, name=name, toName="image", **CONTROL_ATTRS.get(result_type, {}))
            for label in sorted(classes[result_type]):
                ET.SubElement(control, "Label", value=label)
    # ET.indent is Python 3.9+; the XML is valid unindented
    if hasattr(ET, "indent"):
        ET.indent(root)
    return ET.tostring(root, encoding="unicode") + "\n"
