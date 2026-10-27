"""Deprecated CSV label format tools. Convert to json with lmi_utils.label_utils.csv_to_json and use the json tools."""

import importlib
import runpy
import sys
import warnings


def warn_moved(rel_name: str):
    warnings.warn(
        f"lmi_utils.label_utils.{rel_name} moved to {__name__}.{rel_name}. The CSV label format is deprecated: "
        "convert with lmi_utils.label_utils.csv_to_json and use apply_ops, json_to_yolo, json_to_coco or json_to_lst.",
        FutureWarning,
        stacklevel=3,
    )


def redirect(rel_name: str, caller_name: str):
    """Make the old module path an alias of the moved module, or run the moved module when the old path is run as a script."""
    warn_moved(rel_name)
    new_name = f"{__name__}.{rel_name}"
    if caller_name == "__main__":
        runpy.run_module(new_name, run_name="__main__", alter_sys=True)
    else:
        sys.modules[caller_name] = importlib.import_module(new_name)
