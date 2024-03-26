## Documentation

*work in progress*

#### Scripts

- `augment_with_csv.py`: This script augments images and its annotations by applying a random constant to the image. The script takes the following arguments: --path_imgs [IMAGES_FOLDER_PATH] --path_csv [LABELS_CSV_FILE_PATH] --pixel_multiplier [MAX_AUGMENTATION_MULTIPLIER_CONSTANT] --data_size_multipler [NUM_DATA_SAMPLES_TO_GENERATE] --path_out [OUTPUT_FOLDER_PATH]
- `coco_verifier.py`: This script visualizes the dataset and labels for a coco dataset. The script takes the following arguments:
--path_imgs [IMAGES_FOLDER_PATH] --path_json [COCO_JSON_FILE_PATH] --path_out [OUTPUT_FOLDER_PATH]
- `convert_data_to_yolo.py`: This script converts a csv file with annotations to yolo format. The script takes the following arguments: --path_csv [LABELS_CSV_FILE_PATH] --path_imgs [IMAGES_FOLDER_PATH] --path_out [OUTPUT_FOLDER_PATH] --class_map_json [CLASS_MAP_JSON_FILE_PATH] --path_out [OUTPUT_FOLDER_PATH] --seg (whether annotations are part of segmentation or not) --convert (whether to convert bbox-to-mask if --seg else defaults to mask-to-bbox) --bg_images (whether to save images with no labels, yolo models will treat them as background)
- `crop_scale_by_csv.py`: This script crops and scales images and its annotations by using a csv file.
- `crop_scale_labeled_image.py`:
- `csv_to_coco_json.py`:
- `csv_to_json.py`:
- `csv_utils.py`:
- `image_in_label_filter.py`:
- `json_to_ground_truth.py`:
- `label_modify.py`
- `lst_json_to_coco.py`:
- `lst_json_to_csv.py`:
- `mask.py`:
- `pad_with_csv.py`:
- `plot_coco_json.py`:
- `plot_labels.py`:
- `rect.py`:
- `resize_with_csv.py`:
- `rm_background.py`:
- `rot90_with_csv.py`:
- `stretch_to_edges.py`:
- `shape.py`:
- `split_train_test_csv.py`:
- `via_json_to_scv.py`:
- `yolo_txt_to_csv.py`:
