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
6. [Anomaly model files renamed to versioned sub-packages, and AD tiling moved to the pipeline](#6-anomaly-model-files-renamed-to-versioned-sub-packages-and-ad-tiling-moved-to-the-pipeline)
7. [Inference scripts renamed to `infer.py` with shared flags](#7-inference-scripts-renamed-to-inferpy-with-shared-flags)

**New Features**

1. [`preprocess` & `revert_preprocess` — batch processing with history-based reconstruction](#1-preprocess--revert_preprocess--batch-processing-with-history-based-reconstruction)
2. [Forward preprocessing — typed step builders (`lmi_utils.preprocess_utils.steps`)](#2-forward-preprocessing--typed-step-builders)
3. [Revert path — typed history for manual reconstruction](#3-revert-path--typed-history-for-manual-reconstruction)
4. [Tiled object detection](#4-tiled-object-detection)


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

The `operators` parameter is also now formally typed and uses the **unified preprocessing-history schema** — the same shape returned by `Preprocessor.preprocess()` and consumed by `Reconstructor`. See new feature 3 below.

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

## 6. Anomaly model files renamed to versioned sub-packages, and AD tiling moved to the pipeline

**PR:** [#261](../../pull/261)

**What changed:** The three anomaly model files have been reorganised into per-version sub-packages under `anomaly_detectors/anomalib_lmi/`.

| Old path | New path |
|---|---|
| `anomaly_detectors/anomalib_lmi/anomaly_model.py` | `anomaly_detectors/anomalib_lmi/v0/model.py` |
| `anomaly_detectors/anomalib_lmi/anomaly_model2.py` | `anomaly_detectors/anomalib_lmi/v1/model.py` |
| `anomaly_detectors/anomalib_lmi/anomaly_model_v2.py` | `anomaly_detectors/anomalib_lmi/v2/model.py` |

**Tiling is no longer configured on the AD model.** At `v1.6.0`, `AnomalyDetector` read `tile_size`, `stride` and `tile_mode` from the metadata (or from the first three positional arguments) and passed them to the model as `tile=`, `stride=`, `tile_mode=`, which built a `self.tiler` on it. The repackaged backends take `(model_path, **kwargs)` and read only `device` and `image_size`, so those three keys stopped reaching anything. They have now been removed from `AnomalyDetector`. Tile an AD model with the pipeline's `tile` preprocessing step instead, which also stitches the per-tile score maps back to the source image size.

**Before:**
```python
model = AnomalyDetector({"model_path": "model.pt", "tile_size": [300, 300], "stride": [300, 300]})
anom_map = model.predict([img])[0]
```

**After:**
```python
from lmi_utils.preprocess_utils import steps as pre_steps
from lmi_utils.preprocess_utils.preprocessor import Preprocessor
from lmi_utils.preprocess_utils.reconstructor import Reconstructor

model = AnomalyDetector({"model_path": "model.pt"})
step = pre_steps.tile(tile_size=[300, 300], stride=[300, 300])

tiles, history = Preprocessor().preprocess([img], [step])
ad_maps = model.predict(tiles)
anom_map = Reconstructor().reconstruct_images(ad_maps, history)[0]  # back at the source image size
```

`PipelineBase` does the same thing from a manifest — declare a `tile` step in the model's preprocessing and `revert_preprocess()` stitches the score map (see new feature 1).

> **Impact:** All direct imports using the old module paths will raise `ModuleNotFoundError`. `AnomalyDetector` no longer reads `tile_size`, `stride` or `tile_mode` from the metadata, and no longer treats the first three positional arguments as tile settings — with `model_path` in the metadata, positional arguments are ignored with a warning. Note that between the repackaging and their removal these keys were accepted and silently ignored, so an intermediate build ran untiled with no warning: check any AD config that sets them.

---

## 7. Inference scripts renamed to `infer.py` with shared flags

**What changed:** Every model's command-line inference script is now called `infer.py`, and the object detection scripts take the same flags.

| Old command | New command |
|---|---|
| `python -m object_detectors.ultralytics_lmi.yolo.run_model` | `python -m object_detectors.ultralytics_lmi.yolo.infer` |
| `python -m classifiers.ultralytics_lmi.yolo.run_model` | `python -m classifiers.ultralytics_lmi.yolo.infer` |
| `python -m object_detectors.rf_detr_lmi.infer` | unchanged |
| `python -m object_detectors.detectron2_lmi.cli test` | unchanged, and `python -m object_detectors.detectron2_lmi.infer` now works too |

Flags shared by the object detection scripts:

| Flag | Meaning | Replaces |
|---|---|---|
| `-w`, `--weights` | model weights file | `--wts_file` (YOLO) |
| `-i`, `--input` | input image folder | `--path_imgs` (YOLO) |
| `-o`, `--output` | output folder | `--path_out` (YOLO) |
| `-c`, `--confidence` | confidence threshold | `--conf` (RF-DETR), which still works as a short form |
| `-s`, `--image_size` | model input size: one int for a square, or `h w`. Optional: read from the model by default | `--sz h w`, required (YOLO) |
| `--json` | save predictions to `predictions.json` in the output folder, in the LMI dataset json format | `--csv` (YOLO's `preds.csv`); Detectron2's `predictions.csv`, written every run |
| `--tile`, `--stride` | run on tiles and merge the results back; one int for a square, or `h w`. Also saves an image per input to `tiles/` showing the tile grid and each detection colored by how merging built it | new |
| `--no_label` | do not draw class names and scores on the output images | `--no-label` (YOLO only), which still works |
| `--line_thickness` | px width of the drawn boxes and tile grid lines; grows with the image size by default | new |

The classifier script takes `-w/--weights`, `-i/--input`, `-o/--output` and an optional `-s/--image_size`.

**Before:**
```bash
python -m object_detectors.ultralytics_lmi.yolo.run_model -w best.pt -i images -o out --sz 640 640 --csv
python -m object_detectors.rf_detr_lmi.infer -w best.pth -i images -o out --conf 0.5
```

**After:**
```bash
python -m object_detectors.ultralytics_lmi.yolo.infer -w best.pt -i images -o out --json
python -m object_detectors.rf_detr_lmi.infer -w best.pth -i images -o out -c 0.5
```

> **Impact:** Commands that call `run_model`, or pass `--wts_file`, `--path_imgs`, `--path_out` or `--sz`, fail with an error. No script writes a CSV any more: pass `--json` for `predictions.json`. RF-DETR no longer resizes each image to the model input size before drawing, so its output images and predictions are at the original image size.


---

## New Features

### 1. `preprocess` & `revert_preprocess` — batch processing with history-based reconstruction

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

A `tile` step changes the image count, so results come back one entry per source image rather than per tile — see new feature 4.

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

### 2. Forward preprocessing — typed step builders

When you need to apply preprocessing **beyond what the model manifest declares** — e.g. extra flip, resize or crop — build the step list with `lmi_utils.preprocess_utils.steps`.

**Forward step builders:**

| Step builder | Required kwargs | Notes |
|---|---|---|
| `steps.resize(width=..., height=..., preserve_aspect=False, pad_value=0, mode="bilinear")` | — | Each dim defaults to the source image's matching dim. `pad_value` is the letterbox fill, only used when `preserve_aspect=True` (e.g. `114` to match YOLO) |
| `steps.cropbox(boxes=...)` | `boxes` | One `[x1, y1, x2, y2]` per image |
| `steps.flip(lr=False, ud=False)` | — | Defaults to a no-op |
| `steps.pad(width=None, height=None, pad=None, value=0)` | one of `width/height` or `pad` | `pad=[L, R, T, B]` is positive to pad / negative to crop; a `width`/`height` smaller than the input center-crops, otherwise pad |
| `steps.tile(tile_size=..., stride=..., scale_mode="padding", overlap_mode="average")` | `tile_size`, `stride` | Scalars are broadcast to `[h, w]`. Takes further options controlling how tiled detections are merged back — see new feature 4 |

**Usage — inside a `PipelineBase` subclass:**

```python
from lmi_utils.preprocess_utils import steps

class MyPipeline(PipelineBase):
    def predict(self, configs, inputs):
        image = inputs["image"]

        ops = [
            steps.cropbox(boxes=[[100, 50, 900, 700]]),
            steps.resize(width=640, height=640, preserve_aspect=True),
            steps.flip(lr=True),
        ]

        preprocessed, history = self.preprocessor.preprocess(image, ops)
        # preprocessed: list of transformed images, ready for model.predict(...)
        # history:      record of what each step did — feed into the revert path (see new feature 3)
```

---

### 3. Revert path — typed history for manual reconstruction

To map coordinates back to the original image, the revert path needs to know what each preprocessing step did. That record is the **history** — one entry per step. **If the `Preprocessor` did the preprocessing**, you already have the history — just feed it back (the typical flow is in new feature 1).

This section covers the other case: the image was preprocessed **outside** the `Preprocessor` (e.g. cropped by an earlier stage), so no history exists yet. You still want coordinates back in the original space, so you build the history yourself — one entry per step — using the revert step builders below.

**Revert step builders:**

| Step builder | Key fields (per-image lists, length B) |
|---|---|
| `steps.revert_cropbox(boxes=..., orig_sizes=...)` | `boxes` = `[x1, y1, x2, y2]` used; `orig_sizes` = `[W, H]` of the pre-crop canvas |
| `steps.revert_resize(src_sizes=..., dst_sizes=..., pads=...)` | `src_sizes` / `dst_sizes` = `[W, H]`; `pads` = `[L, R, T, B]` letterbox padding |
| `steps.revert_pad(pads=...)` | `pads` = `[L, R, T, B]` applied |
| `steps.revert_flip(lr=..., ud=..., sizes=...)` | `lr`, `ud` flags; `sizes` = `[W, H]` of the flipped image |
| `steps.revert_tile(tile_sizes=..., strides=..., im_sizes=..., scale_sizes=..., n_tiles=..., batch_sizes=..., num_channels=..., scale_modes=..., overlap_modes=...)` | One entry per *source* image |

**Example — image was cropped upstream; revert OD detections into the original frame:**

```python
from lmi_utils.preprocess_utils import steps

# foreground_im was already cropped from the full-resolution image at [x1, y1, x2, y2];
# original canvas was W x H. Build the matching history and let predict() revert for us.
history = [steps.revert_cropbox(boxes=[[x1, y1, x2, y2]], orig_sizes=[[W, H]])]

results, _ = model.predict(foreground_im, 0.5, operators=history)
# results["boxes"][0] is now in the original (W, H) coordinate space.

# Or run inference first and revert afterward — the same history works with revert_preprocess():
results1, _ = model.predict(foreground_im, 0.5)      # coords still in crop space
results2 = self.revert_preprocess(results1, history)
# results2["boxes"][0] is now in the original (W, H) coordinate space.
```

> **Impact:** The revert path is now **typed-only**. Code that built legacy operator dicts must migrate to the revert step builders above — `revert_to_origin`, `revert_mask_to_origin`, `revert_masks_to_origin`, and `apply_operations` keep their names but raise `TypeError` on dict input.

---

### 4. Tiled object detection

Run a detector on overlapping tiles of a large image and get one set of detections back in image coordinates. Useful when objects are small relative to the image, so downscaling the whole frame to the model's input size would lose them.

Supported for boxes, segments and masks. Keypoints and oriented boxes raise.

**The result is per source image, not per tile.** `predict()` is handed N tiles but the tile history folds them back, so every value in the result dict has one entry per *source* image.

There are two ways to run it, differing only in when the merge happens.

**Either let `predict()` revert and merge:**

```python
import numpy as np
import torch

from lmi_utils.preprocess_utils import steps
from lmi_utils.preprocess_utils.preprocessor import Preprocessor

image = torch.from_numpy(np.ascontiguousarray(image)).cuda()  # optional, faster for segmentation models (see below)

step = steps.tile(tile_size=[640, 640], stride=[512, 512])
tiles, history = Preprocessor().preprocess([image], [step])   # e.g. 12 tiles

results, _ = model.predict(tiles, operators=history, configs=0.5)
boxes = results["boxes"][0]   # one entry, not 12 — already in `image` coordinates
```

**Or predict first and revert afterwards**, when you want the per-tile detections too, or the tiles go through something else in between:

```python
from lmi_utils.preprocess_utils.reconstructor import Reconstructor

tiles, history = Preprocessor().preprocess([image], [step])

per_tile, _ = model.predict(tiles, configs=0.5)                        # 12 entries, each in its own tile's coords
results = Reconstructor().reconstruct_coordinates(per_tile, history)   # 1 entry, in image coordinates
```

Inside a `PipelineBase` subclass the same two routes read as `self.preprocess("od-model", image)` followed by either `predict(..., operators=ops_list)` or `self.revert_preprocess(results, ops_list)` — `revert_preprocess()` dispatches a results dict to the `Reconstructor` (see new feature 1).

The results come back in the same form as the image you passed in: numpy arrays for a numpy image, torch tensors on the GPU for a GPU tensor (class names are always numpy). Call `.cpu().numpy()` on them if later code needs numpy.

> **Performance:** after tiling, the tiles' detections must be merged back into one result per image, and for segmentation models this is heavy work on large masks.
> - `predict(..., operators=history)` always merges on the GPU, whatever image you passed in.
> - Reverting afterwards merges on the GPU only if the image was a GPU tensor. With a numpy image it merges on the CPU, which was about twice as slow in our tests.
>
> So if you revert afterwards, convert the image to a GPU tensor first, as in the example above. Box-only models are barely affected either way.

The detector scripts do this for you behind `--tile`/`--stride` (see breaking change 7). `object_detectors.od_core.infer_cli.predict_tiled(model, image, step, **predict_kwargs)` is the first route plus the tile rectangles for plotting.

**Objects split across a seam are rejoined.** A tile only sees part of an object that crosses its edge, so the detector's box stops at the edge. Merging groups those pieces and emits one detection per object: if some tile saw the object whole, that detection wins; if every view is cut, the group's shapes are combined (box, mask or polygon). Class-aware NMS then runs across tiles.

**Merge options** — all on `steps.tile(...)`, applied when coordinates are reverted:

| Option | Default | Meaning |
|---|---|---|
| `merge_fragments` | `None` | `None` merges where the grid allows it and skips where it does not; `True` demands it and raises on a grid that cannot support it; `False` leaves seam-split objects split |
| `score_threshold` | `0.0` | An extra threshold on top of the per-class `configs` confidence the model already applied, dropping detections *before* merging so a weak piece cannot represent its group and take the whole group down with it. Off by default |
| `nms_iou` | `0.5` | Class-aware NMS IoU across tiles; `None` disables both NMS rules |
| `containment` | `0.8` | Share of one detection that must lie inside another to count as contained; `None` disables the containment rule |
| `edge_tolerance` | `2.0` | Px from a tile edge that still counts as touching it. Absolute, not a fraction of the tile: it tracks the detector's box-regression error at a crop boundary. Results change little between `0.5` and `4` |
| `min_label_size` | `0.0` | Forward direction only: drop a clipped *label* thinner than this many px on either axis when projecting ground truth into tiles |
| `report_merge_origin` | `False` | Add a `merge_origin` code per detection saying how it was built |

Merging needs `scale_mode="padding"` and more than `2 * edge_tolerance` px of overlap on both axes. With the default tolerance that means a stride at least 5 px shorter than the tile.

The detector scripts turn `report_merge_origin` on and colour each box by how merging built it, in the plot they write to `tiles/` in the output folder. That plot is the quickest way to see whether a grid is merging the way you expect. The codes themselves are the `ORIGIN_*` constants in `lmi_utils.postprocess_utils.tile_merge`.

**Training on tiles.** `apply_ops tile` cuts a labeled dataset into the same grid, clipping each label into every tile it reaches:

```bash
python -m lmi_utils.label_utils.apply_ops -i images -oi tiled_images \
    tile --width 640 --height 640 --stride 512 --min_label_size 8
```

Each tile becomes its own file carrying `source_id`, so tiles of one image stay on the same side of a train/val split. Tiles with no labels are dropped unless you pass `--bg`.
