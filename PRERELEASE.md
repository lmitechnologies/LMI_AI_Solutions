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
9. [Unified preprocessing-history schema — shared by `Preprocessor`, OD `predict(operators=…)`, and the legacy `revert_to_origin` helpers](#9-unified-preprocessing-history-schema)
10. [Typed step builders — `lmi_utils.preprocess_utils.steps`](#10-typed-step-builders--lmi_utilspreprocess_utilssteps)


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

### 8. `crop-to-label` preprocessing step with runtime channel

`preprocess()` now accepts an optional `runtime` argument: a `{id: value}` dict that supplies caller-side data which isn't known until inference time. Each key must match the `id` of a manifest step. The first consumer is the new `crop-to-label` step, which declares "this model expects a crop around region `<label>`" in the manifest and gets the actual per-image box at runtime from an upstream detector. `revert_preprocess()` automatically maps coordinates back through the crop offset.

Supported step types are now: `resize`, `tile`, `crop-to-label`.

**Two-stage pipeline: foreground detector → defect detector on the bottle crop**

Manifest declares the crop intent on the defect model. The `id` field on the crop step is what the runtime dict will key against:
```json
"bottom-defect": {
    "details": {
        "preprocessing": [
            {"type": "crop-to-label", "id": "b50c1466-377d-436c-a594-a06c00397f7b", "configuration": {"label": "BOTTLE-BBOX"}},
            {"type": "resize", "id": "e9d8c7b6-a5f4-4e3d-2c1b-0a9f8e7d6c5b", "configuration": {"height": 640, "width": 640, "preserve_aspect": true}}
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

# 2. Defect detector — pass the bottle box through the runtime channel,
#    keyed by the crop step's manifest `id`.
bottle_box = fg_out["boxes"][0][0].tolist()        # [x1, y1, x2, y2]
runtime = {"b50c1466-377d-436c-a594-a06c00397f7b": {"boxes": [bottle_box]}}

def_in, def_hist = self.preprocess("bottom-defect", image, runtime=runtime)
def_out, _ = self.models["bottom-defect"].predict(def_in, 0.5)
def_out = self.revert_preprocess(def_out, def_hist)  # offsets added back automatically
```

When a model has multiple `crop-to-label` steps, the runtime dict keys route per step by `id` (e.g. `{"f1a2b3c4-d5e6-4f78-9a0b-1c2d3e4f5a6b": {...}, "0a9b8c7d-6e5f-4a3b-2c1d-0e9f8a7b6c5d": {...}}`).

---

### 9. Unified preprocessing-history schema

The metadata returned by `Preprocessor.preprocess()`, accepted by OD `predict(operators=…)`, and consumed by `revert_to_origin` / `revert_mask_to_origin` / `revert_masks_to_origin` / `apply_operations` now uses **one** shape. The legacy single-key dicts (`{"resize": [tw, th, ow, oh]}`, `{"pad": [L, R, T, B]}`, etc.) have been replaced by named per-image metadata.

Users are **not** expected to construct these dicts by hand — `self.preprocessor.preprocess()` returns the history in this shape, and you pass it straight back to `revert_preprocess()` or `predict(operators=…)`. The schema below is documented for reference and for the rare manual-construction case.

**Canonical schema:**

```python
# Each history entry:
{
    "type": "<op_name>",                  # e.g. "resize", "pad", "crop", "flip", "tile"
    "metadata": [<per_image_dict>, ...],  # length 1 (broadcast to batch) or B (per-image)
    "id": "<optional>",                    # propagated from the manifest step when present
}
```

**Per-op metadata fields (per-image):**

| `type` | `metadata` dict |
|---|---|
| `resize` | `{"src_size": [w, h], "dst_size": [w, h], "pad"?: [L, R, T, B]}` — `pad` present only when `preserve_aspect=True` produced letterbox padding |
| `pad` | `{"pad": [L, R, T, B]}` |
| `crop` | `{"box": [x1, y1, x2, y2], "orig_size": [w, h]}` |
| `flip` | `{"lr": bool, "ud": bool, "size": [w, h]}` |
| `rotate` | `{"angle": float, "src_size": [w, h], "dst_size": [w, h]}` |
| `tile` | `{"tile_size": [h, w], "stride": [h, w], "im_size": [H, W], "scale_size": [H', W'], "n_tiles": [n_h, n_w], "batch_size": int, "num_channel": int, "scale_mode": str, "overlap_mode": str}` |

**Before (legacy):**

```python
# Single chain (legacy revert_to_origin / od_base operators)
operators = [
    {"resize": [640, 480, 1280, 960]},   # [tw, th, ow, oh]
    {"pad":    [0, 0, 80, 80]},          # [L, R, T, B]
]

# Per-image chains
operators = [
    [{"resize": [640, 640, 1280, 720]}],
    [{"resize": [640, 640, 1024, 768]}],
]

```

**After (one unified shape everywhere):**

```python
# Same shape for: history, operators, and manual-construction.
operators = [
    {
        "type": "resize",
        "metadata": [{"src_size": [1280, 960], "dst_size": [640, 480]}],
    },
    {
        "type": "pad",
        "metadata": [{"pad": [0, 0, 80, 80]}],
    },
]

# Per-image batch — just lengthen `metadata`:
operators = [
    {
        "type": "resize",
        "metadata": [
            {"src_size": [1280, 720], "dst_size": [640, 640]},
            {"src_size": [1024, 768], "dst_size": [640, 640]},
        ],
    },
]

```

**Manual construction (no `Preprocessor` needed):**

```python
# Cropped foreground manually, want to revert detections back into original image space:
x1, y1, x2, y2 = bottle_bbox
crop_op = {"type": "crop", "metadata": [{"box": [x1, y1, x2, y2], "orig_size": [W, H]}]}

# Pass directly to predict() or to revert_to_origin / revert_masks_to_origin:
results = model.predict(foreground_im, 0.5, operators=[crop_op])
```

> **Impact:** Any code that builds legacy-shape `{op_name: [positional_list]}` operator dicts must migrate to the new schema. The legacy helpers (`revert_to_origin`, `revert_mask_to_origin`, `revert_masks_to_origin`, `apply_operations`) keep their names but only accept the new shape — feeding legacy dicts raises errors`. 

---

### 10. Typed step builders — `lmi_utils.preprocess_utils.steps`

For **extra or manual preprocessing** beyond what the model manifest declares, `lmi_utils.preprocess_utils.steps` provides typed builders for each step. Each builder returns a dict in the same shape as a manifest entry (`{"type": str, "configuration": dict, "id"?: str}`), so it plugs directly into `Preprocessor.preprocess()`.

**Available builders:**

| Builder | Step `type` | Required kwargs | Notes |
|---|---|---|---|
| `steps.resize(*, width=None, height=None, preserve_aspect=False, mode="bilinear", id=None)` | `resize` | `width` | `height` optional; `None` values are stripped from `configuration` |
| `steps.crop(*, boxes, id=None)` | `crop` | `boxes` | `boxes` is a list of `[x1, y1, x2, y2]` |
| `steps.crop_to_label(*, label, id=None)` | `crop-to-label` | `label` | Pair with `runtime={id: {"boxes": [...]}}` at inference (see section 8) |
| `steps.flip(*, lr=False, ud=False, id=None)` | `flip` | — | Defaults to a no-op (both `False`) |
| `steps.pad(*, width=None, height=None, pad=None, value=0, id=None)` | `pad` | one of `width/height` or `pad` | `pad=[L, R, T, B]` for explicit padding |
| `steps.rotate(*, angle, id=None)` | `rotate` | `angle` | Degrees, positive = clockwise; canvas is expanded to fit the rotated extent |
| `steps.tile(*, tile_size, stride, scale_mode="padding", overlap_mode="average", id=None)` | `tile` | `tile_size`, `stride` | Scalars are accepted and broadcast to `[h, w]` |

Build an ad-hoc ops list and pass it to `self.preprocessor.preprocess(images, ops)` inside a `PipelineBase` subclass:

```python
from lmi_utils.preprocess_utils import steps

class MyPipeline(PipelineBase):
    def predict(self, configs, inputs):
        image = inputs["image"]

        ops = [
            steps.crop_to_label(label="BOTTLE-BBOX", id="bottle_crop"),
            steps.resize(width=640, height=640, preserve_aspect=True),
            steps.flip(lr=True),
        ]
        runtime = {"bottle_crop": {"boxes": [[100, 50, 900, 700]]}}

        preprocessed, history = self.preprocessor.preprocess(image, ops, runtime=runtime)
        # preprocessed: list of transformed images
        # history: unified preprocessing-history list, ready for predict(operators=history) or self.revert_preprocess(results, history).
```
