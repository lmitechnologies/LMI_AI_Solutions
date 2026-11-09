# Pre-Release Notes — v2.1.0 → next

Breaking changes and new features since `v2.1.0`. Each breaking change says what to update.

## Breaking Changes

- **Label JSON leaves out unset keypoint fields** ([#481](../../pull/481)): a keypoint's `visibility` and `bounding_box_id` are
  omitted when unset, instead of written as `null`, which GoFactory parsed as an unknown annotation. Code that reads label JSON
  must treat a missing key as unset, e.g. `data.get("visibility")`.

## New Features

- **YOLO26** ([#482](../../pull/482)): `model_name: yolov26` is registered for od, seg, obb and pose, on the same backends as
  `yolov8` and `yolov11`.
- **Keypoint uploads from pipelines** ([#481](../../pull/481)): `PipelineBase.add_prediction("boxes", ..., keypoints=, model_role=)`
  adds the box plus one keypoint per slot, linked by `bounding_box_id` and named from the model's keypoint names. A third keypoint
  column is read as confidence. `YoloPose` adds `keypoint_layout` (class name → slot names) and `n_kpts`, and warns when the model
  has no keypoint names.
