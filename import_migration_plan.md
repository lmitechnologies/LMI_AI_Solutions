# Import Migration Plan: Flattened Namespace → Fully-Qualified Imports

## Problem Summary

The current `pyproject.toml` uses `where` to promote sub-packages from 4 directories into the **global top-level namespace**:

```toml
[tool.setuptools.packages.find]
where = ["anomaly_detectors", "classifiers", "lmi_utils", "object_detectors"]
```

This means sub-folders like `label_utils`, `image_utils`, `ad_core`, etc. are importable as if they were top-level packages — hiding which parent component they actually belong to.

---

## Why This Is Problematic

### 1. Namespace Collision Risk 🔴
All sub-packages from 4 different `where` directories share the global namespace. Generic names like `image_utils`, `label_utils`, or `eval_utils` could easily collide with:
- Third-party PyPI packages
- Other internal projects
- Future sub-packages added to any of the 4 `where` directories

### 2. Ambiguous Ownership 🔴
When reading `from label_utils.shapes import Rect`, there is no indication that `label_utils` lives inside `lmi_utils/`. Developers (and tools) cannot determine the parent component without consulting the file tree.

### 3. Cross-Component Coupling 🟡
Scripts freely import across component boundaries without any visible hierarchy:
- `object_detectors/yolov8_lmi/run_model.py` → imports `label_utils`, `gadget_utils` (from `lmi_utils/`)
- `lmi_utils/eval_utils/benchmark_AD.py` → imports `anomalib_lmi` (from `anomaly_detectors/`)
- `lmi_utils/pipeline_base/pipeline_base.py` → imports `ad_core`, `cls_core` (from `anomaly_detectors/`, `classifiers/`)

### 4. Fragile Without Editable Install 🟡
Running any script directly (e.g. `python lmi_utils/eval_utils/benchmark_AD.py`) fails with `ModuleNotFoundError` because the flattened paths are only available after `pip install -e .`.

---

## Current Import Map

Below is the complete mapping of **which sub-packages are imported from where**:

### Sub-packages under `lmi_utils/`
| Sub-package | Imported by files in | Import count |
|---|---|---|
| `label_utils` | `lmi_utils/`, `object_detectors/`, `tests/` | ~50 |
| `image_utils` | `lmi_utils/`, `anomaly_detectors/`, `object_detectors/`, `tests/` | ~23 |
| `gadget_utils` | `lmi_utils/`, `anomaly_detectors/`, `object_detectors/`, `classifiers/`, `tests/` | ~16 |
| `dataset_utils` | `lmi_utils/`, `object_detectors/`, `tests/` | ~29 |
| `system_utils` | `lmi_utils/`, `tests/` | ~10 |
| `preprocess_utils` | `lmi_utils/`, `tests/` | ~7 |
| `pcl_utils` | `lmi_utils/` | ~7 |
| `eval_utils` | `lmi_utils/` | ~2 |
| `postprocess_utils` | `object_detectors/` | ~1 |
| `pipeline_base` | `lmi_utils/` | ~2 |
| `data_utils` | root `temp.py` | ~1 |

### Sub-packages under `anomaly_detectors/`
| Sub-package | Imported by files in | Import count |
|---|---|---|
| `anomalib_lmi` | `anomaly_detectors/`, `lmi_utils/`, `tests/` | ~10 |
| `ad_core` | `anomaly_detectors/`, `lmi_utils/`, `tests/` | ~8 |

### Sub-packages under `classifiers/`
| Sub-package | Imported by files in | Import count |
|---|---|---|
| `cls_core` | `classifiers/`, `lmi_utils/`, `tests/` | ~4 |
| `yolov8_cls` | `tests/` | ~1 |

### Sub-packages under `object_detectors/`
| Sub-package | Imported by files in | Import count |
|---|---|---|
| `yolov8_lmi` | `object_detectors/`, `tests/` | ~5 |
| `yolov5_lmi` | `object_detectors/` | ~3 |
| `ultralytics_lmi` | `tests/` | ~1 |
| `detectron2_lmi` | `tests/` | ~1 |
| `rf_detr_lmi` | `object_detectors/` | ~1 |

**Total: ~180+ import statements** that need updating.

---

## Target State

### `pyproject.toml` change

```toml
[tool.setuptools.packages.find]
where = ["."]
include = [
    "anomaly_detectors*",
    "classifiers*",
    "lmi_utils*",
    "object_detectors*",
]
exclude = [
    "*submodules*",
    "*legacy*",
    "*deprecated*",
    "*tf_objdet*",
]
```

### Import style change

```python
# BEFORE (flattened — current)
from label_utils.shapes import Rect
from ad_core.anomaly_detector import AnomalyDetector
from gadget_utils.pipeline_utils import plot_one_box
import gadget_utils.pipeline_utils as pipeline_utils

# AFTER (fully-qualified — target)
from lmi_utils.label_utils.shapes import Rect
from anomaly_detectors.ad_core.anomaly_detector import AnomalyDetector
from lmi_utils.gadget_utils.pipeline_utils import plot_one_box
import lmi_utils.gadget_utils.pipeline_utils as pipeline_utils
```

### Entry points change

```toml
# BEFORE
[project.scripts]
labels-preprocess = "label_utils.apply_ops:main"
labels-convert-yolo = "label_utils.json_to_yolo:main"
labels-convert-lst = "label_utils.lst_to_json:main"
gadget-surface = "gadget_utils.gadget_surface_utils:main"
gadget-image = "gadget_utils.gadget_image:main"

# AFTER
[project.scripts]
labels-preprocess = "lmi_utils.label_utils.apply_ops:main"
labels-convert-yolo = "lmi_utils.label_utils.json_to_yolo:main"
labels-convert-lst = "lmi_utils.label_utils.lst_to_json:main"
gadget-surface = "lmi_utils.gadget_utils.gadget_surface_utils:main"
gadget-image = "lmi_utils.gadget_utils.gadget_image:main"
```

---

## Migration Phases

### Phase 1: `lmi_utils` internal imports (lowest risk)
**Scope**: Files under `lmi_utils/` that import other `lmi_utils/` sub-packages.

These are intra-component imports, so the change is straightforward: prefix every import with `lmi_utils.`.

| File | Change |
|---|---|
| `lmi_utils/eval_utils/pr_curve.py` | `label_utils` → `lmi_utils.label_utils` |
| `lmi_utils/eval_utils/iou_from_csv2.py` | `label_utils` → `lmi_utils.label_utils` |
| `lmi_utils/eval_utils/benchmark_AD.py` | `preprocess_utils` → `lmi_utils.preprocess_utils` |
| `lmi_utils/dataset_utils/representations.py` | `image_utils` → `lmi_utils.image_utils`, `gadget_utils` → `lmi_utils.gadget_utils`, `dataset_utils` → `lmi_utils.dataset_utils` |
| `lmi_utils/dataset_utils/file_utils.py` | `dataset_utils` → `lmi_utils.dataset_utils` |
| `lmi_utils/dataset_utils/ops/*.py` | `dataset_utils` → `lmi_utils.dataset_utils`, `image_utils` → `lmi_utils.image_utils`, `gadget_utils` → `lmi_utils.gadget_utils` |
| `lmi_utils/image_utils/*.py` | `gadget_utils` → `lmi_utils.gadget_utils`, `image_utils` → `lmi_utils.image_utils`, `system_utils` → `lmi_utils.system_utils` |
| `lmi_utils/label_utils/*.py` (many files) | `label_utils` → `lmi_utils.label_utils`, `image_utils` → `lmi_utils.image_utils`, `gadget_utils` → `lmi_utils.gadget_utils`, etc. |
| `lmi_utils/pipeline_base/pipeline_base.py` | `ad_core` → `anomaly_detectors.ad_core`, `cls_core` → `classifiers.cls_core`, `dataset_utils` → `lmi_utils.dataset_utils`, `preprocess_utils` → `lmi_utils.preprocess_utils` |
| `lmi_utils/gadget_utils/profile_to_hmap.py` | `image_utils` → `lmi_utils.image_utils` |
| `lmi_utils/pcl_utils/*.py` | `pcl_utils` → `lmi_utils.pcl_utils`, `image_utils` → `lmi_utils.image_utils` |
| `lmi_utils/preprocess_utils/handlers/*.py` | `gadget_utils` → `lmi_utils.gadget_utils`, `image_utils` → `lmi_utils.image_utils` |

**Estimated changes**: ~80 import lines across ~30 files

---

### Phase 2: `anomaly_detectors` internal + cross-component imports
**Scope**: Files under `anomaly_detectors/` and files that import from `anomaly_detectors/`.

| File | Change |
|---|---|
| `anomaly_detectors/anomalib_lmi/anomaly_model.py` | `ad_core` → `anomaly_detectors.ad_core`, `gadget_utils` → `lmi_utils.gadget_utils` |
| `anomaly_detectors/anomalib_lmi/anomaly_model2.py` | Same pattern |
| `anomaly_detectors/anomalib_lmi/anomaly_model_v2.py` | Same pattern + `image_utils` → `lmi_utils.image_utils` |
| `anomaly_detectors/anomalib_lmi/base.py` | `gadget_utils` → `lmi_utils.gadget_utils`, `anomalib_lmi` → `anomaly_detectors.anomalib_lmi` |
| `anomaly_detectors/anomalib_lmi/gofactory_AD.py` | `anomalib_lmi` → `anomaly_detectors.anomalib_lmi` |
| `anomaly_detectors/anomalib_lmi/convert_to_torchscript.py` | `anomalib_lmi` → `anomaly_detectors.anomalib_lmi` |
| `anomaly_detectors/anomalib_lmi/ad_utils.py` | `anomalib_lmi` → `anomaly_detectors.anomalib_lmi` |
| `lmi_utils/eval_utils/benchmark_AD.py` | `anomalib_lmi` → `anomaly_detectors.anomalib_lmi` |

**Estimated changes**: ~20 import lines across ~8 files

---

### Phase 3: `object_detectors` and `classifiers` imports
**Scope**: Files under `object_detectors/` and `classifiers/`, plus their cross-component imports.

| File | Change |
|---|---|
| `object_detectors/yolov8_lmi/run_model.py` | `gadget_utils` → `lmi_utils.gadget_utils`, `label_utils` → `lmi_utils.label_utils`, `yolov8_lmi` → `object_detectors.yolov8_lmi` |
| `object_detectors/yolov8_lmi/model.py` | `gadget_utils` → `lmi_utils.gadget_utils` |
| `object_detectors/yolov5_lmi/*.py` | Same pattern |
| `object_detectors/ultralytics_lmi/yolo/*.py` | Same pattern |
| `object_detectors/detectron2_lmi/*.py` | Same + `postprocess_utils` → `lmi_utils.postprocess_utils` |
| `object_detectors/rf_detr_lmi/*.py` | `gadget_utils` → `lmi_utils.gadget_utils`, `dataset_utils` → `lmi_utils.dataset_utils` |
| `object_detectors/gofactory/*.py` | `dataset_utils` → `lmi_utils.dataset_utils` |
| `classifiers/yolov8_cls/model.py` | `cls_core` → `classifiers.cls_core` |
| `classifiers/yolov8_cls/run_model.py` | `gadget_utils` → `lmi_utils.gadget_utils` |

**Estimated changes**: ~30 import lines across ~12 files

---

### Phase 4: Tests
**Scope**: All files under `tests/`.

| File | Change |
|---|---|
| `tests/lmi_utils/label_utils/test_bbox_utils.py` | `label_utils` → `lmi_utils.label_utils` |
| `tests/lmi_utils/image_utils/test_img_resize.py` | `image_utils` → `lmi_utils.image_utils` |
| `tests/lmi_utils/image_utils/test_img_tile.py` | `image_utils` → `lmi_utils.image_utils`, `system_utils` → `lmi_utils.system_utils` |
| `tests/lmi_utils/image_utils/test_tiler.py` | Same pattern |
| `tests/lmi_utils/dataset_utils/test_representations.py` | `dataset_utils` → `lmi_utils.dataset_utils` |
| `tests/lmi_utils/preprocess_utils/test_*.py` | `preprocess_utils` → `lmi_utils.preprocess_utils` |
| `tests/anomaly_detectors/anomalib_lmi/test_*.py` | `ad_core` → `anomaly_detectors.ad_core`, `anomalib_lmi` → `anomaly_detectors.anomalib_lmi`, `gadget_utils` → `lmi_utils.gadget_utils` |
| `tests/classifiers/yolov8_cls/test_*.py` | `cls_core` → `classifiers.cls_core`, `yolov8_cls` → `classifiers.yolov8_cls` |
| `tests/object_detectors/*/test_*.py` | `yolov8_lmi` → `object_detectors.yolov8_lmi`, `ultralytics_lmi` → `object_detectors.ultralytics_lmi`, `detectron2_lmi` → `object_detectors.detectron2_lmi` |

**Estimated changes**: ~30 import lines across ~12 files

---

### Phase 5: `pyproject.toml` and final cleanup
1. Update `[tool.setuptools.packages.find]` to use `where = ["."]` with `include`.
2. Update all `[project.scripts]` entry points.
3. Re-install with `pip install -e .` and verify.
4. Run full test suite.

---

## Automated Migration Script

A `sed`/PowerShell script can handle the bulk of the renaming. The core substitutions are:

```
# lmi_utils sub-packages (add "lmi_utils." prefix)
label_utils       →  lmi_utils.label_utils
image_utils       →  lmi_utils.image_utils
gadget_utils      →  lmi_utils.gadget_utils
dataset_utils     →  lmi_utils.dataset_utils
eval_utils        →  lmi_utils.eval_utils
system_utils      →  lmi_utils.system_utils
preprocess_utils  →  lmi_utils.preprocess_utils
postprocess_utils →  lmi_utils.postprocess_utils
pipeline_base     →  lmi_utils.pipeline_base
pcl_utils         →  lmi_utils.pcl_utils
data_utils        →  lmi_utils.data_utils

# anomaly_detectors sub-packages (add "anomaly_detectors." prefix)
ad_core           →  anomaly_detectors.ad_core
anomalib_lmi      →  anomaly_detectors.anomalib_lmi

# classifiers sub-packages (add "classifiers." prefix)
cls_core          →  classifiers.cls_core
yolov8_cls        →  classifiers.yolov8_cls

# object_detectors sub-packages (add "object_detectors." prefix)
yolov8_lmi        →  object_detectors.yolov8_lmi
yolov5_lmi        →  object_detectors.yolov5_lmi
ultralytics_lmi   →  object_detectors.ultralytics_lmi
detectron2_lmi    →  object_detectors.detectron2_lmi
rf_detr_lmi       →  object_detectors.rf_detr_lmi
gofactory         →  object_detectors.gofactory
od_core           →  object_detectors.od_core
```

> [!CAUTION]
> The automated script must only match import lines (`from X` or `import X` patterns) to avoid accidentally replacing non-import occurrences of these names (e.g. in strings, comments, or file paths).

---

## Verification

After each phase:
1. Run `pip install -e .` to re-install the package.
2. Run the full test suite: `pytest tests/`.
3. Verify entry points work: `labels-preprocess --help`.
4. Run `ruff check .` to ensure no lint regressions.

---

## Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Downstream consumers use the old import paths | Announce breaking change in release notes; bump major version |
| Dockerfiles or CI scripts reference old paths | Search Dockerfiles and CI configs for affected imports |
| Jupyter notebooks use old imports | Search `.ipynb` files (e.g. `gadget_utils/notebooks/`) |
| `__init__.py` re-exports break | Review each `__init__.py` for re-exports that need updating |

---

## Timeline Estimate

| Phase | Effort | Risk |
|---|---|---|
| Phase 1: lmi_utils internal | ~2 hours | Low |
| Phase 2: anomaly_detectors | ~1 hour | Low |
| Phase 3: object_detectors + classifiers | ~1 hour | Medium |
| Phase 4: Tests | ~1 hour | Low |
| Phase 5: pyproject.toml + cleanup | ~30 min | Medium |
| **Total** | **~5-6 hours** | |
