# Pre-Release Notes — v2.2.0 → next

Breaking changes and new features since `v2.2.0`. Each breaking change says what to update.

## Breaking Changes

- **A keypoint names its box in `parent_id`**: label JSON, `KeypointAnnotation`, and pipeline predictions carry a keypoint's box
  as `parent_id` instead of `bounding_box_id`, a name that can hold other containment relationships later. GoFactory reads only
  `parent_id`. Rename the key in label JSON written by earlier versions, e.g.
  `find <dataset> -name '*.json' -exec sed -i 's/"bounding_box_id"/"parent_id"/g' {} +`, and in code that reads or writes it.
