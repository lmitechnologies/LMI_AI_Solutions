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
8. [Forward preprocessing — typed step builders (`lmi_utils.preprocess_utils.steps`)](#8-forward-preprocessing--typed-step-builders)
9. [Revert path — typed history for manual reconstruction](#9-revert-path--typed-history-for-manual-reconstruction)


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

The `operators` parameter is also now formally typed and uses the **unified preprocessing-history schema** — the same shape returned by `Preprocessor.preprocess()` and consumed by `Reconstructor`. See section 9 below.

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

When an OD model's preprocessed input still doesn't match its training size, the pipeline auto-injects a final resize. A letterbox injection pads with the model's `RESIZE_PAD_VALUE` (`114` for YOLO, `0` otherwise) to match training-time padding. A manifest-declared `resize` step pads with `0` unless its configuration sets `pad_value`.

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
            {"type": "resize", "id": "a1f2c3d4-5e6f-4a7b-8c9d-0e1f2a3b4c5d", "configuration": {"height": 640, "width": 640, "preserve_aspect": true}}
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
            {"type": "resize", "id": "7b2d4e6f-8a9c-4b1d-9e3f-5a6b7c8d9e0f", "configuration": {"height": 224, "width": 448, "preserve_aspect": true}},
            {"type": "tile", "id": "c3e5f7a9-1b2d-4c6e-8f0a-2b4d6f8a0c1e", "configuration": {"height": 224, "width": 224, "y_stride": 112, "x_stride": 112}}
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

### 8. Forward preprocessing — typed step builders

When you need to apply preprocessing **beyond what the model manifest declares** — e.g. extra flip, resize or crop — build the step list with `lmi_utils.preprocess_utils.steps`.

**Forward step builders:**

| Step builder | Required kwargs | Notes |
|---|---|---|
| `steps.resize(width=..., height=..., preserve_aspect=False, pad_value=0, mode="bilinear")` | — | Each dim defaults to the source image's matching dim. `pad_value` is the letterbox fill, only used when `preserve_aspect=True` (e.g. `114` to match YOLO) |
| `steps.crop(boxes=...)` | `boxes` | One `[x1, y1, x2, y2]` per image |
| `steps.flip(lr=False, ud=False)` | — | Defaults to a no-op |
| `steps.pad(width=None, height=None, pad=None, value=0)` | one of `width/height` or `pad` | `pad=[L, R, T, B]` for explicit padding |
| `steps.tile(tile_size=..., stride=..., scale_mode="padding", overlap_mode="average")` | `tile_size`, `stride` | Scalars are broadcast to `[h, w]` |

**Usage — inside a `PipelineBase` subclass:**

```python
from lmi_utils.preprocess_utils import steps

class MyPipeline(PipelineBase):
    def predict(self, configs, inputs):
        image = inputs["image"]

        ops = [
            steps.crop(boxes=[[100, 50, 900, 700]]),
            steps.resize(width=640, height=640, preserve_aspect=True),
            steps.flip(lr=True),
        ]

        preprocessed, history = self.preprocessor.preprocess(image, ops)
        # preprocessed: list of transformed images, ready for model.predict(...)
        # history:      record of what each step did — feed into the revert path (see section 9)
```

---

### 9. Revert path — typed history for manual reconstruction

To map coordinates back to the original image, the revert path needs to know what each preprocessing step did. That record is the **history** — one entry per step. **If the `Preprocessor` did the preprocessing**, you already have the history — just feed it back (the typical flow is in section 7).

This section covers the other case: the image was preprocessed **outside** the `Preprocessor` (e.g. cropped by an earlier stage), so no history exists yet. You still want coordinates back in the original space, so you build the history yourself — one entry per step — using the revert step builders below.

**Revert step builders:**

| Step builder | Key fields (per-image lists, length B) |
|---|---|
| `steps.revert_crop(boxes=..., orig_sizes=...)` | `boxes` = `[x1, y1, x2, y2]` used; `orig_sizes` = `[W, H]` of the pre-crop canvas |
| `steps.revert_resize(src_sizes=..., dst_sizes=..., pads=...)` | `src_sizes` / `dst_sizes` = `[W, H]`; `pads` = `[L, R, T, B]` letterbox padding |
| `steps.revert_pad(pads=...)` | `pads` = `[L, R, T, B]` applied |
| `steps.revert_flip(lr=..., ud=..., sizes=...)` | `lr`, `ud` flags; `sizes` = `[W, H]` of the flipped image |
| `steps.revert_tile(tile_sizes=..., strides=..., im_sizes=..., scale_sizes=..., n_tiles=..., batch_sizes=..., num_channels=..., scale_modes=..., overlap_modes=...)` | One entry per *source* image |

**Example — image was cropped upstream; revert OD detections into the original frame:**

```python
from lmi_utils.preprocess_utils import steps

# foreground_im was already cropped from the full-resolution image at [x1, y1, x2, y2];
# original canvas was W x H. Build the matching history and let predict() revert for us.
history = [steps.revert_crop(boxes=[[x1, y1, x2, y2]], orig_sizes=[[W, H]])]

results, _ = model.predict(foreground_im, 0.5, operators=history)
# results["boxes"][0] is now in the original (W, H) coordinate space.

# Or run inference first and revert afterward — the same history works with revert_preprocess():
results1, _ = model.predict(foreground_im, 0.5)      # coords still in crop space
results2 = self.revert_preprocess(results1, history)
# results2["boxes"][0] is now in the original (W, H) coordinate space.
```

> **Impact:** The revert path is now **typed-only**. Code that built legacy operator dicts must migrate to the revert step builders above — `revert_to_origin`, `revert_mask_to_origin`, `revert_masks_to_origin`, and `apply_operations` keep their names but raise `TypeError` on dict input.
