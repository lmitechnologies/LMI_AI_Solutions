# Pre-Release Notes — v1.6.0 → next

This document covers every **breaking change** introduced after the `v1.6.0` tag. Each section identifies what changed, which PR introduced it, and a concrete before/after comparison.

---

## Table of Contents

1. [Installation: unified single-package install](#1-installation-unified-single-package-install)
2. [Import paths: fully-qualified module paths required](#2-import-paths-fully-qualified-module-paths-required)
3. [Removed submodules: efficientnet, tf_objdet, PaddleOCR](#3-removed-submodules-efficientnet-tf_objdet-paddleocr)
4. [YOLO v0 support dropped; yolov8_lmi/yolov8_cls deprecated](#4-yolo-v0-support-dropped-yolov8_lmiyolov8_cls-deprecated)
5. [OD predict() — batched output and renamed parameters](#5-od-predict--batched-output-and-renamed-parameters)
6. [Results.to_dict() — parameter renamed and semantics inverted](#6-resultsto_dict--parameter-renamed-and-semantics-inverted)
7. [Results.classes type change and Results.numpy() removal](#7-resultsclasses-type-change-and-resultsnumpy-removal)
8. [AD predict() — now returns a list and supports batch_size](#8-ad-predict--now-returns-a-list-and-supports-batch_size)
9. [Anomaly model files renamed to versioned sub-packages](#9-anomaly-model-files-renamed-to-versioned-sub-packages)
10. [model_shape attribute removed from anomaly models](#10-model_shape-attribute-removed-from-anomaly-models)
11. [Preprocessor — new signature, returns (images, history)](#11-preprocessor--new-signature-returns-images-history)
12. [Preprocessor handlers — new signature and metadata key](#12-preprocessor-handlers--new-signature-and-metadata-key)
13. [Box.to_mask() — mask_type argument dropped](#13-boxto_mask--mask_type-argument-dropped)

---

## 1. Installation: unified single-package install

**PRs:** [#183](../../pull/183) (consolidate pyproject.toml), [#219](../../pull/219) (auto package discovery)

**What changed:** The four separate installable packages (`lmi_utils`, `object_detectors`, `classifiers`, `anomaly_detectors`) have been merged into a single root package `lmi_ai_solutions`. All sub-directory `pyproject.toml` files have been removed. Package discovery is now automatic.

**Before (v1.6.0):**
```bash
# Install each sub-package separately
pip install -e lmi_utils/
pip install -e object_detectors/
pip install -e classifiers/
pip install -e anomaly_detectors/
```

**After:**
```bash
# Install everything from the repo root
pip install -e .
```

> **Impact:** Any CI/CD pipeline, Dockerfile, or developer setup that installs sub-packages individually will break. Update all install steps to use the single root install.

---

## 2. Import paths: fully-qualified module paths required

**PR:** [#207](../../pull/207)

**What changed:** Short-form bare imports (relying on `PYTHONPATH` entries pointing at subdirectories) no longer work. All imports must use the full package path from the repo root.

**Before:**
```python
from image_utils.tiler import OverlapMode, ScaleMode, Tiler
from dataset_utils.representations import Dataset
from gadget_utils.pipeline_utils import get_img_path_batches
from ad_core.anomaly_detector_registry import AnomalyDetectorRegistry
from anomalib_lmi.anomaly_model import AnomalyModel
from cls_core.classifier_registry import ClassifierRegistry
```

**After:**
```python
from lmi_utils.image_utils.tiler import OverlapMode, ScaleMode, Tiler
from lmi_utils.dataset_utils.representations import Dataset
from lmi_utils.gadget_utils.pipeline_utils import get_img_path_batches
from anomaly_detectors.ad_core.anomaly_detector_registry import AnomalyDetectorRegistry
from anomaly_detectors.anomalib_lmi.v0.model import AnomalyModel
from classifiers.cls_core.classifier_registry import ClassifierRegistry
```

Also, `lmi_ai.env` has been simplified — it no longer adds every subdirectory to `PYTHONPATH`. Re-sourcing the new `lmi_ai.env` is required.

> **Impact:** Every script that uses bare sub-package imports will raise `ModuleNotFoundError` at runtime.

---

## 3. Removed submodules: efficientnet, tf_objdet, PaddleOCR

**PR:** [#218](../../pull/218), [#226](../../pull/226)

**What changed:** Three git submodules and the `ocr_models/` folder have been permanently removed:
- `classifiers/efficientnet`
- `object_detectors/tf_objdet/models`
- `ocr_models/PaddleOCR`

**Before:**
```bash
git submodule update --init --recursive   # populated efficientnet, tf_objdet, PaddleOCR
```
```python
# these worked in v1.6.0
from classifiers.efficientnet import ...
from object_detectors.tf_objdet import ...
from ocr_models.PaddleOCR import ...
```

**After:**
```python
# ImportError — these packages no longer exist
```

> **Impact:** Any code using EfficientNet classification, TensorFlow object detection, or PaddleOCR must be migrated off this repo. The `.gitmodules` file no longer references these submodules.

---

## 4. YOLO v0 support dropped; yolov8_lmi/yolov8_cls deprecated

**PR:** [#224](../../pull/224)

**What changed:**
- The `v0` YOLO model variant is no longer registered and cannot be instantiated.
- `object_detectors/yolov8_lmi/` and `classifiers/yolov8_cls/` have been moved to their respective `deprecated/` subdirectories.
- A new `classifiers/ultralytics_lmi/` package is introduced as the replacement for YOLO classification.

**Before (detection):**
```python
from object_detectors.yolov8_lmi.model import Yolov8

model = Yolov8(model_path="yolov8n.pt", version="v0")
```

**After:**
```python
from object_detectors.ultralytics_lmi.yolo.model import UltralyticsYolo

model = UltralyticsYolo(model_path="yolov8n.pt")   # v0 option removed
```

**Before (classification):**
```python
from classifiers.yolov8_cls.model import Yolov8_cls
```

**After:**
```python
from classifiers.ultralytics_lmi.yolo.model import UltralyticsYoloCls
```

> **Impact:** The `version="v0"` constructor argument is silently ignored or raises an error. Any import from `yolov8_lmi` or `yolov8_cls` at their old paths will still work for now (they are in `deprecated/`) but will be removed in a future release.

---

## 5. OD predict() — batched output and renamed parameters

**PR:** [#250](../../pull/250)

**What changed:** `predict()` is now implemented in `ODBase` and shared by all backends. The return value changed from a single-image result dict to a **batched** result dict where every value is a list (one entry per image).

**Before:**
```python
results, time_info = model.predict(image, configs=0.5)

# results was a flat dict for a single image
boxes   = results["boxes"]    # shape (N, 4)
scores  = results["scores"]   # shape (N,)
classes = results["classes"]  # list[str]
```

**After:**
```python
results, time_info = model.predict(image, configs=0.5)

# results is now batched — index [0] to get the single-image data
boxes   = results["boxes"][0]    # shape (N, 4)
scores  = results["scores"][0]   # shape (N,)
classes = results["classes"][0]  # list[str]
```

**Batch usage (new capability):**
```python
images = [img1, img2, img3]
results, time_info = model.predict(images, configs=0.5)

for boxes, scores, classes in zip(results["boxes"], results["scores"], results["classes"]):
    ...  # process per image
```

The `operators` parameter is also now formally typed:
- `None` — no coordinate reversion (unchanged).
- `list[dict]` — one operator chain applied to all images.
- `list[list[dict]]` — per-image operator chains (must match batch size).

---

## 6. Results.to_dict() — parameter renamed and semantics inverted

**PR:** [#250](../../pull/250)

**What changed:** The `return_tensor` parameter of `Results.to_dict()` was renamed to `return_numpy` and its boolean semantics were **inverted**.

**Before:**
```python
d = results.to_dict(return_tensor=True)   # returns torch.Tensor values
d = results.to_dict(return_tensor=False)  # returns numpy arrays
```

**After:**
```python
d = results.to_dict(return_numpy=False)   # returns torch.Tensor values (default)
d = results.to_dict(return_numpy=True)    # returns numpy arrays
```

> **Impact:** Passing `return_tensor=True` will silently be ignored (unexpected keyword argument in some Python versions, or treated as unrecognized kwarg). Code that relied on `return_tensor=False` to get numpy arrays must pass `return_numpy=True`.

---

## 7. Results.classes type change and Results.numpy() removal

**PR:** [#250](../../pull/250)

**What changed:**
- `Results.classes` is now `np.ndarray` (dtype `str_`) instead of `list[str]`.
- `Results.numpy()` method has been removed.

**Before:**
```python
results.classes          # list[str], e.g. ["cat", "dog"]
results.numpy()          # returned a new Results with all tensors as numpy
```

**After:**
```python
results.classes          # np.ndarray, e.g. array(["cat", "dog"], dtype="<U3")
results.to_dict(return_numpy=True)  # use this to get numpy arrays instead
```

> **Impact:** Code that calls `.numpy()` directly on a `Results` object will raise `AttributeError`. Code checking `isinstance(results.classes, list)` will break.

---

## 8. AD predict() — now returns a list and supports batch_size

**PR:** [#263](../../pull/263)

**What changed:** `AnomalyDetector.predict()` (and all `ADBase` subclasses) now return a **list** of per-image anomaly maps instead of a single array. A `batch_size` kwarg is added for chunked inference.

**Before:**
```python
ad_score = model.predict(image)  # np.ndarray [H, W]
```

**After:**
```python
ad_scores = model.predict(image)      # list[np.ndarray], length 1 for a single image
ad_score  = model.predict(image)[0]   # index [0] to recover previous behaviour

# batch usage
ad_scores = model.predict([img1, img2, img3], batch_size=2)  # list of 3 maps
```

The `predict()` signature is now:
```python
def predict(self, image: ImageBatch, **kwargs) -> List[ImageLike]:
    ...
    # kwargs:
    #   batch_size (int): chunk size for mini-batch inference
```

> **Impact:** Any code that assigns the return value of `predict()` to a single array and then indexes it directly (e.g., `score[y, x]`) will fail with a `TypeError` or unexpected result. Always index `[0]` after calling predict with a single image.

---

## 9. Anomaly model files renamed to versioned sub-packages

**PR:** [#261](../../pull/261)

**What changed:** The three anomaly model files have been reorganised into per-version sub-packages under `anomaly_detectors/anomalib_lmi/`.

| Old path | New path |
|---|---|
| `anomaly_detectors/anomalib_lmi/anomaly_model.py` | `anomaly_detectors/anomalib_lmi/v0/model.py` |
| `anomaly_detectors/anomalib_lmi/anomaly_model2.py` | `anomaly_detectors/anomalib_lmi/v1/model.py` |
| `anomaly_detectors/anomalib_lmi/anomaly_model_v2.py` | `anomaly_detectors/anomalib_lmi/v2/model.py` |

**Before:**
```python
from anomaly_detectors.anomalib_lmi.anomaly_model  import AnomalyModel
from anomaly_detectors.anomalib_lmi.anomaly_model2 import AnomalyModel2
from anomaly_detectors.anomalib_lmi.anomaly_model_v2 import AnomalyModelV2
```

**After:**
```python
from anomaly_detectors.anomalib_lmi.v0.model import AnomalyModel
from anomaly_detectors.anomalib_lmi.v1.model import AnomalyModel2
from anomaly_detectors.anomalib_lmi.v2.model import AnomalyModelV2
```

Config files moved analogously (e.g., `configs/v1/patchcore.yaml` → `v1/configs/patchcore.yaml`).

> **Impact:** All direct imports using the old module paths will raise `ModuleNotFoundError`.

---

## 10. model_shape attribute removed from anomaly models

**PR:** [#265](../../pull/265)

**What changed:** The `model_shape` attribute on anomaly model instances (set after loading a TRT engine or TorchScript model) has been removed. Use `image_size` instead.

**Before:**
```python
model.warmup(model_path="model.engine", image_size=[256, 256])
h, w = model.model_shape   # [256, 256]
```

**After:**
```python
model.warmup(model_path="model.engine", image_size=[256, 256])
h, w = model.image_size    # [256, 256]
```

> **Impact:** Any code accessing `model.model_shape` will raise `AttributeError`.

---

## 11. Preprocessor — new signature, returns (images, history)

**PRs:** [#180](../../pull/180), [#269](../../pull/269)

**What changed:** `Preprocessor.preprocess()` now accepts a list (or batch) of images and returns a `(processed_images, history)` tuple. The old single-image in / single-image out interface is gone. Gadget version < 2.4 is no longer supported.

**Before:**
```python
from lmi_utils.preprocess_utils.preprocessor import Preprocessor

p = Preprocessor()
steps = [{"type": "resize", "configuration": {"size": [640, 640], "keep_aspect_ratio": True}}]

out_image = p.preprocess(image, steps)   # np.ndarray [H, W, C]
```

**After:**
```python
from lmi_utils.preprocess_utils.preprocessor import Preprocessor

p = Preprocessor()
steps = [{"type": "resize", "configuration": {"size": [640, 640], "keep_aspect_ratio": True}}]

# Single image — wrap in list, unwrap the result
out_images, history = p.preprocess([image], steps)
out_image = out_images[0]    # np.ndarray [H, W, C]

# history is now used for reconstruction; each entry looks like:
# {"type": "resize", "metadata": [<per-image ops list>]}
```

`Preprocessor` also now accepts a BHWC numpy/tensor array directly and handles 2-D (HW) grayscale images.

---

## 12. Preprocessor handlers — new signature and metadata key

**PRs:** [#180](../../pull/180), [#269](../../pull/269)

**What changed:** Custom handler functions and the built-in `revert_resize` / `revert_tile` functions have new signatures. The metadata dict key returned by handlers changed from step-specific names (`"ops"`, `"tiler_metadata"`) to a unified `"metadata"` key. The history record key also changed from `"configuration"` to `"metadata"`.

**Handler function signature — before:**
```python
def my_handler(image: np.ndarray, **kwargs) -> np.ndarray:
    ...
    return processed_image
```

**Handler function signature — after:**
```python
def my_handler(images: List[torch.Tensor], config: Dict[str, Any]) -> Tuple[List[torch.Tensor], Dict]:
    ...
    # metadata dict MUST contain a "metadata" key
    return processed_images, {"metadata": per_image_metadata_list}
```

**revert_resize — before:**
```python
revert_resize(images, meta={"ops": per_image_ops_list})
```

**revert_resize — after:**
```python
revert_resize(images, metadata=per_image_ops_list)   # pass the list directly
```

**revert_tile — before:**
```python
revert_tile(images, meta={"tiler_metadata": tiler_meta_list})
```

**revert_tile — after:**
```python
revert_tile(images, metadata=tiler_meta_list)   # pass the list directly
```

**History record key — before:**
```python
# entry returned by Preprocessor.preprocess()
{"type": "resize", "configuration": {"ops": [...]}}
```

**History record key — after:**
```python
{"type": "resize", "metadata": [...]}   # flattened; "configuration" wrapper removed
```

---

## 13. Box.to_mask() — mask_type argument dropped

**PR:** [#240](../../pull/240)

**What changed:** `Box.to_mask()` no longer accepts the `mask_type` kwarg. It always returns a binary `Mask`. To get a `Polygon` representation of a box, call the new `to_polygon()` method.

**Before:**
```python
from lmi_utils.dataset_utils.representations import AnnotationType

mask = box.to_mask(h=480, w=640, mask_type=AnnotationType.MASK)
poly = box.to_mask(h=480, w=640, mask_type=AnnotationType.POLYGON)
```

**After:**
```python
mask = box.to_mask(h=480, w=640)   # always returns Mask; mask_type kwarg silently ignored or errors
poly = box.to_polygon(h=480, w=640)
```

`h` and `w` are now **required** keyword arguments. Calling `to_mask()` without them raises `ValueError`.

---

## Potential breaking changes not covered above

The following are changes that do not carry a `!` commit marker but may still break existing code depending on usage:

| Area | Change | PR |
|---|---|---|
| **Tiler decoupled from AD model** | `Tiler` is no longer embedded inside anomaly model subclasses. Callers that relied on tiling being done internally by the model must now orchestrate it separately via `Preprocessor`. | [#263](../../pull/263) |
| **OD `configs` arg is now positional** | `predict(image, configs, operators)` — `configs` was previously a keyword arg in several backends. | [#250](../../pull/250) |
| **`Results` always has boxes/scores/classes** | `Results.to_dict()` now always includes `boxes`, `scores`, and `classes` even when there are no detections (empty tensors/arrays). Code checking for key absence will behave differently. | [#250](../../pull/250) |
| **`Results.is_seg` flag** | `Results.__init__` now accepts `is_seg=True` to unconditionally include `masks`/`segments` in `to_dict()`. Positional construction of `Results` may break if not updated. | [#250](../../pull/250) |
| **Gadget pipeline schema** | `PipelineBase` schema v2 changed to require global preprocessing config. Pipelines targeting gadget < 2.4 will fail schema validation. | [#180](../../pull/180) |
| **OCR models removed** | The `ocr_models/` directory has been deleted entirely. There is no migration path within this repo. | [#226](../../pull/226) |
| **AD `annotate()` GPU path** | `ADBase.annotate()` now uses a GPU-accelerated turbo colormap LUT on `self.device`. Code that calls `annotate()` on CPU-only machines should still work (tensor falls back to CPU), but timings and dtype assumptions may differ. | [#263](../../pull/263) |
| **`Preprocessor.preprocess()` validates 4D input** | A BHWC numpy/tensor array is now accepted directly. Code that manually looped over a batch before calling `preprocess()` should still work but the batch can now be passed directly. | [#269](../../pull/269) |
| **`annotate()` silently clamps `ad_max`** | If `ad_max <= ad_threshold`, `annotate()` now adjusts `ad_max` to `ad_threshold + 1e-8` automatically rather than producing a corrupted heatmap. Code that deliberately passed equal values to suppress the heatmap will no longer work as intended. | [#179](../../pull/179) |
| **`representations.py` classes converted to `@dataclass`** | `Box`, `Polygon`, `Mask`, `Label`, `Point2d` are now dataclasses. Positional and keyword construction still works, but subclasses overriding `__init__`, code using `vars(box)` or `box.__dict__`, and `isinstance` checks against the old class identity may behave differently. | [#238](../../pull/238) |
| **`ModelFactory` import path** | `ModelFactory` was introduced after v1.6.0 at `object_detectors.od_core.model_factory`, then moved to `lmi_common.model_factory` in [#262](../../pull/262). Any code written against the intermediate import path must update to `from lmi_common.model_factory import ModelFactory`. | [#250](../../pull/250), [#262](../../pull/262) |
