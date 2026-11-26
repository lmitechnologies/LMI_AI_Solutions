# Pre-Release Notes — v2.2.0 → next

Breaking changes and new features since `v2.2.0`. Each breaking change says what to update.

## Breaking Changes

- **A keypoint names its box in `parent_id`**: label JSON, `KeypointAnnotation`, and pipeline predictions carry a keypoint's box
  as `parent_id` instead of `bounding_box_id`, a name that can hold other containment relationships later. GoFactory reads only
  `parent_id`. Rename the key in label JSON written by earlier versions, e.g.
  `find <dataset> -name '*.json' -exec sed -i 's/"bounding_box_id"/"parent_id"/g' {} +`, and in code that reads or writes it.

## New Features

- **GoFactory classifiers load**: the classifier registry accepts the `ultralytics8` framework that GoFactory sets as
  `training_package` (`Ultralytics8`), as the object detector registry already does.
- **Class folders to Factory**: `python -m lmi_utils.label_utils.class_folders_to_factory -i <dataset> -o <factory_dataset>` converts a
  classification dataset laid out as one folder per class (optionally inside `train`/`val`/`test`) to a Factory dataset.
  `--link` hard-links the images instead of copying them, and `--per_class N` keeps a random sample of each class.
