# Pre-Release Notes — v1.6.0 → next

This document covers all **breaking changes and new features** introduced after the `v1.6.0` tag. Each section identifies what changed, which PR introduced it, and a concrete before/after comparison.

---

## Table of Contents

**Breaking Changes**
1. [Installation: unified single-package install](#1-installation-unified-single-package-install)
2. [Import paths: fully-qualified module paths required](#2-import-paths-fully-qualified-module-paths-required)
3. [OD predict() — batched output and renamed parameters](#3-od-predict--batched-output-and-renamed-parameters)
4. [AD predict() — now returns a list and supports batch_size](#4-ad-predict--now-returns-a-list-and-supports-batch_size)
5. [YOLO v0 support dropped](#5-yolo-v0-support-dropped)
6. [Anomaly model files renamed to versioned sub-packages](#6-anomaly-model-files-renamed-to-versioned-sub-packages)

**New Features**

7. [Preprocessor & Reconstructor — batch processing with history-based reconstruction](#7-preprocessor--reconstructor--batch-processing-with-history-based-reconstruction)
8. [`crop-to-label` preprocessing step with runtime channel](#8-crop-to-label-preprocessing-step-with-runtime-channel)


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
from gadget_utils.pipeline_utils import revert_to_origin
from ultralytics_lmi.yolo.model import Yolo
```

**After:**
```python
from lmi_utils.gadget_utils.pipeline_utils import revert_to_origin
from object_detectors.ultralytics_lmi.yolo.model import Yolo
```

Also, `lmi_ai.env` has been simplified — it no longer adds every subdirectory to `PYTHONPATH`. Re-sourcing the new `lmi_ai.env` is required.

> **Impact:** Every script that uses bare sub-package imports will raise `ModuleNotFoundError` at runtime.

---

## 3. OD predict() — batched output and renamed parameters

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

## 4. AD predict() — now returns a list and supports batch_size

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

## 5. YOLO v0 support dropped

**PR:** [#224](../../pull/224)

**What changed:** The `v0` YOLO model variant is no longer registered and cannot be instantiated.

**Before:**
```python
model = ObjectDetector(
    metadata=dict(
        version="v0",
        model_name="yolov8",
        task=task,
        framework="ultralytics",
        model_path=path,
        image_size=image_size,
    ),
    device=device,
)
```

**After:**
```python
model = ObjectDetector(
    metadata=dict(
        version="v1",
        model_name="yolov8",
        task=task,
        framework="ultralytics",
        model_path=path,
        image_size=image_size,
    ),
    device=device,
)
```

> **Impact:** Passing `version="v0"` will raise an error. Update all instantiations to `version="v1"`.

---

## 6. Anomaly model files renamed to versioned sub-packages

**PR:** [#261](../../pull/261)

**What changed:** The three anomaly model files have been reorganised into per-version sub-packages under `anomaly_detectors/anomalib_lmi/`.

| Old path | New path |
|---|---|
| `anomaly_detectors/anomalib_lmi/anomaly_model.py` | `anomaly_detectors/anomalib_lmi/v0/model.py` |
| `anomaly_detectors/anomalib_lmi/anomaly_model2.py` | `anomaly_detectors/anomalib_lmi/v1/model.py` |
| `anomaly_detectors/anomalib_lmi/anomaly_model_v2.py` | `anomaly_detectors/anomalib_lmi/v2/model.py` |


> **Impact:** All direct imports using the old module paths will raise `ModuleNotFoundError`.


---

## New Features

### 7. `preprocess` & `revert_preprocess` — batch processing with history-based reconstruction

**PRs:** [#180](../../pull/180), [#269](../../pull/269)

`PipelineBase` now exposes `preprocess()` and `revert_preprocess()` as the standard preprocessing pair. `preprocess()` accepts a single image, a list of images, or a BHWC array and returns `(processed_images, history)`. `revert_preprocess()` uses the history to invert the operations — restoring original resolution for images (AD) or reverting coordinates back to the original image space (OD).

Supported step types: `resize`, `tile`. Steps can be chained and nested (e.g. resize → tile → tile).

**Resize with Object Detection**

OD model role from gofactory:
```json
"od-model": {
    "format": "pt",
    "details": {
        "classes": ["defect_a"],
        "confidence_threshold": 0.5,
        "image_size": [640, 640],
        "preprocessing": [
            {"type": "resize", "configuration": {"height": 640, "width": 640, "preserve_aspect": true}}
        ],
        "training_package": "Ultralytics",
        "training_algorithm": "Yolo"
    },
    "artifacts": {"pt": {"attributes": {}, "model_path": model_path}},
    "model_role": "od-model",
    "model_name": "od-model",
    "model_type": "InstanceSegmentation",
    "model_version": "1"
}
```

Pipeline example:
```python
from lmi_utils.pipeline_base.pipeline_base import PipelineBase

class MyODPipeline(PipelineBase):
    def load(self, model_roles, configs):
        self.load_models(model_roles, configs, device=device)

    def predict(self, configs, inputs):
        image = inputs["image"]  # single HWC numpy image

        # 1. Preprocess
        preprocessed, ops_list = self.preprocess("od-model", image)  # preprocessed is a list of images

        # 2. Inference
        results1, _ = self.models["od-model"].predict(preprocessed, 0.5)

        # 3. Revert coordinates to original image space
        results2 = self.revert_preprocess(results1, ops_list)

        # 4. Annotate
        r = {k: v[0] for k, v in results2.items()}  # remove the batch dim
        annotated = self.models["od-model"].annotate_image(r, image)

```

**Tiling with anomaly detection**

`Tiler` is no longer embedded inside anomaly model subclasses ([#263](../../pull/263)). Tiling must now be orchestrated explicitly via `preprocess()` before calling `predict()`, and `revert_preprocess()` stitches the per-tile anomaly maps back into a full-resolution map.

AD model role from gofactory:
```json
"ad-model": {
    "format": "pt",
    "details": {
        "image_size": [224, 224],
        "min_threshold": 0.0,
        "max_threshold": 1.0,
        "preprocessing": [
            {"type": "resize", "configuration": {"height": 224, "width": 448, "preserve_aspect": true}},
            {"type": "tile", "configuration": {"height": 224, "width": 224, "y_stride": 112, "x_stride": 112}}
        ],
        "training_package": "Anomalib1",
        "training_algorithm": "Patchcore"
    },
    "artifacts": {"pt": {"attributes": {}, "model_path": model_path}},
    "model_role": "ad-model",
    "model_name": "ad-model",
    "model_type": "AnomalyDetection",
    "model_version": "1"
}
```

Pipeline example:
```python
from lmi_utils.pipeline_base.pipeline_base import PipelineBase

class MyADPipeline(PipelineBase):
    def load(self, model_roles, configs):
        self.load_models(model_roles, configs, device=device)

    def predict(self, configs, inputs):
        image = inputs["image"]

        # 1. Preprocess: resize -> tile
        preprocessed_image, ops_list = self.preprocess("ad-model", image)

        # 2. Inference on tiles
        scores = self.models["ad-model"].predict(preprocessed_image)

        # 3. reconstruct the score with the same shape as image
        final_scores = self.revert_preprocess(scores, ops_list)  # returns a list of images
        final_score = final_scores[0]

```

---

### 8. `crop-to-label` preprocessing step with runtime channel

`preprocess()` now accepts an optional `runtime` argument: a list of step patches keyed by `(type, instance)` that supply caller-side data which isn't known until inference time. The first consumer is the new `crop-to-label` step, which declares "this model expects a crop around region `<label>`" in the manifest and gets the actual per-image box at runtime from an upstream detector. `revert_preprocess()` automatically maps coordinates back through the crop offset.

Supported step types are now: `resize`, `tile`, `crop-to-label`.

**Two-stage pipeline: foreground detector → defect detector on the bottle crop**

Manifest declares the crop intent on the defect model:
```json
"bottom-defect": {
    "details": {
        "preprocessing": [
            {"type": "crop-to-label", "configuration": {"label": "BOTTLE-BBOX"}},
            {"type": "resize", "configuration": {"height": 640, "width": 640, "preserve_aspect": true}}
        ],
        ...
    },
    ...
}
```

Pipeline supplies the box from the foreground model at runtime:
```python
# 1. Foreground detector (no runtime needed)
fg_in, fg_hist = self.preprocess("bottom-foreground", image)
fg_out, _ = self.models["bottom-foreground"].predict(fg_in, 0.5)
fg_out = self.revert_preprocess(fg_out, fg_hist)   # boxes now in original image space

# 2. Defect detector — pass the bottle box through the runtime channel
bottle_box = fg_out["boxes"][0][0].tolist()        # [x1, y1, x2, y2]
runtime = [{"type": "crop-to-label", "runtime": {"boxes": [bottle_box]}}]

def_in, def_hist = self.preprocess("bottom-defect", image, runtime=runtime)
def_out, _ = self.models["bottom-defect"].predict(def_in, 0.5)
def_out = self.revert_preprocess(def_out, def_hist)  # offsets added back automatically
```

Use the optional `instance` field on both the manifest step and the runtime patch when a single model has multiple crop-to-label steps.
