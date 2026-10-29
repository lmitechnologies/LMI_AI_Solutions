# Pre-Release Notes — v2.0.1 → next

Breaking changes and new features since `v2.0.1`. Each breaking change says what to update.

## Breaking Changes

### `lst_to_json` links keypoints to their boxes

`lst_to_json` now sets each annotated keypoint's `bounding_box_id`: the box it is nested under in Label Studio's Regions panel, else the one box that contains it. It fails, listing every problem, when a keypoint lies in no box or in several, is nested under something other than an upright box, or repeats a name already on its box. Nest those keypoints under their boxes, or pass `--keep_unlinked_keypoints` to write them unlinked as before, for datasets whose keypoints are unused or are linked later by `json_to_yolo --target_classes`. Prediction keypoints that cannot be linked only print a warning.

`json_to_lst` writes the links back as Label Studio nesting, so keypoints keep their boxes through a round trip.

### `lst_to_json` keeps fractional pixels

`lst_to_json` writes box corners and polygon points as Label Studio gives them, instead of truncating them to whole pixels, so converted labels shift by up to a pixel from earlier conversions. `json_to_yolo` pose exports now reject a keypoint linked to a rotated target box, or unlinked within a rotated target box's extent, the same rule `lst_to_json` applies; before, rotated boxes were treated as upright.

An upright box drawn as a mask (`json_to_yolo --seg`) covers the pixels whose centers are inside it; it used to truncate its edges, which only differs for fractional boxes. Padding and cropping (`apply_ops pad`) drop a mask only when no pixel of it is left; a mask whose pixels all sat in the first row or column was dropped before. Polygons and rotated boxes are cut at the image edge instead of having their outside corners moved onto it, which kept shapes lying wholly outside and distorted shapes crossing an edge at an angle; a polygon the edge cuts apart becomes one label per piece (`<id>_p0`, `<id>_p1`, …), as tiling names them. `dataset_pad.clip_shapes` now returns the clipped shapes instead of the ids to delete.

`apply_ops crop-by-label` cuts the other labels at the crop's edge the same way, so boxes and polygons outside the crop are dropped instead of kept with corners moved onto its edge or with inverted coordinates. It now keeps keypoints instead of failing on them; a keypoint linked to the crop box is unlinked. The crop covers the pixels whose centers are inside the box, the same pixels as its mask. A rotated crop box of any angle fails, as do a crop label that is not a box and one that covers no pixel. `crop_dataset_by_label` drops its `clip` argument, and its helpers `crop_kp`, `crop_bbox`, `crop_mask` and `crop_by_percent` are removed.

## Deprecations

### CSV label tools

**PR:** [#468](https://github.com/lmitechnologies/LMI_AI_Solutions/pull/468)

The CSV label format is deprecated in favor of the json format (`labels.json`). The CSV tools moved to `lmi_utils.label_utils.csv_labels`. Their old paths, such as `python -m lmi_utils.label_utils.resize_with_csv`, still work but print a `FutureWarning`, and will be removed in a later release.

| Deprecated | Use instead |
|---|---|
| `lst_to_csv` | `lst_to_json` |
| `csv_to_yolo` | `json_to_yolo` |
| `csv_to_coco` | `json_to_coco` |
| `csv_to_lst` | `json_to_lst` |
| `resize_with_csv`, `pad_with_csv`, `rot90_with_csv` | `apply_ops resize`, `apply_ops pad`, `apply_ops rotate` |
| `plot_with_csv` | `plot_with_json` |
| `csv_utils`, `shapes` | `lmi_utils.dataset_utils.representations` |

`augment_hf`, `crop_by_foreground`, `crop_scale_by_csv`, `image_in_label_filter`, `replace_labels`, `rm_background`, `split_train_test_in_csv`, `via_json_to_csv` and `yolo_to_csv` have no json version yet. They keep working; convert their CSV output with `csv_to_json`.

The common Label Studio → YOLO chain becomes:

```bash
# before
python -m lmi_utils.label_utils.lst_to_csv -i $input_path -o $input_path
python -m lmi_utils.label_utils.resize_with_csv -i $input_path -o /app/data/resized --width $W --height $H
python -m lmi_utils.label_utils.csv_to_yolo -i /app/data/resized -o /app/data/resized_yolo --seg

# after
python -m lmi_utils.label_utils.lst_to_json -i $input_path -imgs $input_path -of $input_path/labels.json
python -m lmi_utils.label_utils.apply_ops -i $input_path -oi /app/data/resized resize --width $W --height $H
python -m lmi_utils.label_utils.json_to_yolo -ti /app/data/resized -o /app/data/resized_yolo --seg
```

An existing `labels.csv` converts with `python -m lmi_utils.label_utils.csv_to_json --csv labels.csv -i <images> -o labels.json`.

### detectron2 `csv_to_coco` dataset layout

The detectron2 trainer now reads the `json_to_coco` layout: `_annotations.coco.json` with the images beside it. The `csv_to_coco` layout, `annotations.json` with an `images/` folder, still trains but prints a `FutureWarning`, and will be removed together with the CSV tools.
