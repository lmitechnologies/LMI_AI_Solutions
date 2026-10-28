# Pre-Release Notes — v2.0.1 → next

Breaking changes and new features since `v2.0.1`. Each breaking change says what to update.

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
