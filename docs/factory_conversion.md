# Converting datasets to Factory

Factory imports one directory format.
COCO, YOLO and Label Studio datasets reach it through the AIS dataset json, which every `label_utils` converter already reads and writes:

```
COCO ─────────coco_to_json──┐
YOLO ─────────yolo_to_json──┼──> labels.json ──json_to_factory──> Factory directory
Label Studio ──lst_to_json──┘
```

Keeping the json in the middle means the existing tools (`json_to_yolo`, `json_to_coco`, `apply_ops`, `plot_with_json`) apply to a converted dataset without further work, and Factory has exactly one writer.

## Commands

```bash
# COCO
python3 -m lmi_utils.label_utils.coco_to_json -j annotations.json -i images/
python3 -m lmi_utils.label_utils.json_to_factory -i images/ -o factory_dataset/

# YOLO
python3 -m lmi_utils.label_utils.yolo_to_json -y dataset.yaml
python3 -m lmi_utils.label_utils.json_to_factory -i <dataset root> -o factory_dataset/

# Label Studio
python3 -m lmi_utils.label_utils.lst_to_json -i export.json -imgs images/ -of images/labels.json -ps source_dataset/
python3 -m lmi_utils.label_utils.json_to_factory -i images/ -o factory_dataset/

# Label Studio, when no pose schema exists yet: draft one, edit it, then convert with it
python3 -m lmi_utils.label_utils.lst_to_json -i export.json -imgs images/ -of images/labels.json --scaffold_schema draft.json
```

The COCO and YOLO converters write `labels.json` where `json_to_factory` looks for it by default, so the second command needs no `-j`.

## What Factory needs for pose

A pose dataset's keypoint layout is a *declaration*, not a tally of its annotations.
Factory reads it from the root `.meta.json` that `json_to_factory` writes:

```json
{
  "annotationSchema": {
    "type": "Pose",
    "version": 1,
    "coordinateDimensions": 3,
    "classes": {
      "bolt": {
        "keypoints": ["head", "left-flange", "right-flange"],
        "horizontalFlip": [0, 2, 1],
        "skeleton": [[0, 1], [0, 2]]
      }
    }
  }
}
```

- `keypoints` is the class's tensor order; its length is that class's K. A slot no image observes stays declared.
- `horizontalFlip` maps each slot to the slot it becomes under a mirror. It must be a permutation and its own inverse, since flipping twice has to restore every keypoint. `null` imports fine but blocks `fliplr`/`flipud` during training.
- `skeleton` is visualization metadata, zero-based.
- Two classes may reuse a keypoint name; the owning box scopes it.

Annotations carry the layout by name: each keypoint is one annotation whose `label_id` is the keypoint name and whose `bounding_box_id` links it to its instance.
`json_to_factory` resolves that link, falling back to box containment for a keypoint that has none, and rejects one that no box or several boxes contain.

The declaration is carried on `Label` in the dataset json (`keypoints`, `horizontal_flip`, `skeleton`) and `coordinate_dimensions` on `Dataset`, so it survives the intermediate file.

## COCO specifics

- `categories[].name` becomes the Factory class id. `--class_map` supplies a different mapping.
- `categories[].keypoints` becomes the class layout; `categories[].skeleton` becomes `skeleton`.
- Skeleton indices are read as one-based, the COCO convention. Pass `--skeleton_base 0` for a file that writes them zero-based.
- An instance's `keypoints` triplets are read in the category's declared order. A slot with visibility 0 is dropped, leaving the declared slot empty rather than putting a keypoint at the origin.
- **COCO states no flip symmetry.** Without `--flip_map` every class declares `horizontalFlip: null`. Supply a json mapping each class id to its flip, as keypoint names or local indices:

  ```json
  { "bolt": ["head", "right-flange", "left-flange"] }
  ```

- `--segmentation` converts instances of classes declaring no keypoints to polygons or bitmasks instead of boxes.

## YOLO specifics

- `names` becomes the class ids, `kpt_shape` the slot count and coordinate dimensions.
- A YOLO pose model has one global slot layout, so a class owning fewer slots pads the rest. `kpt_names`
  names each class's slots and may key them by numeric class index, as Ultralytics does, or by class name, as
  Factory's exporter does. Factory padding is marked `__unused_*`; those slots are dropped from the class's
  declared layout.
- Without `kpt_names` the file states one layout for the whole model, which is the only sound reading of it: every class gets all K slots, named by `--keypoint_names` or positionally.
- `flip_idx` is global. It is translated into each class's local order, and a class whose mirror leaves the slots it owns declares no flip — a partial mapping would drop keypoints under a flip rather than mirror them.
- A zeroed slot is unobserved and produces no annotation.
- Every split the yaml declares is converted into one dataset, with each image keeping its path relative to the dataset root so same-named images in different splits stay distinct.
- A stale `path` in the yaml (it records where the dataset was written) falls back to the dataset root.

## Label Studio specifics

- The export must be the full `JSON` format, not `JSON-MIN`, which drops the fields the converter needs.
- **A Label Studio export states no keypoint layout.** Its labeling configuration declares one flat keypoint vocabulary for the whole project, with no per-class order, flip or skeleton, so the declaration has to be supplied: `-ps` takes the source Factory dataset directory, its `.meta.json`, or a bare `annotationSchema` object. It is optional -- without it the export still converts, keypoint labels become classes of their own, and no `.meta.json` is written, which is a dataset Factory can import but not train pose on.
- Where to get the schema: a project Factory created already has one, on the annotation project and on the dataset it came from (the project id is in each task's image URL). For a project built by hand, `--scaffold_schema` writes a draft from the observed annotations to edit and pass back as `-ps`. The draft is a starting point, not a contract -- it groups keypoints under the box class that contains them, in the order the annotator worked, and always leaves `horizontalFlip: null`, because which slot mirrors which is knowledge about the object and appears nowhere in the data.
- With `-ps` given, a label the schema declares as a keypoint becomes a slot of its class rather than a class of its own, and a keypoint naming no declared slot is rejected.
- Native Label Studio keypoint nesting links a keypoint to its box with `parentID`. Older generic relation records
  are accepted as a fallback, in either direction, but cannot override `parentID`. Region ids repeat across
  completions, so ownership is resolved within the completion that declares it. A keypoint with neither form
  falls back to containment, which is ambiguous when boxes overlap.
- **Label Studio allows a keypoint that belongs to no box, which a pose model cannot train on.** `json_to_factory --unlinked_keypoints` decides what becomes of one: `error` (the default, rejecting the dataset), `drop`, or `keep` it in Factory unlinked.
- Label Studio has no visibility concept, so keypoints arrive without one and Factory reads them as visible.
- Images are matched by the name the export gives them, then by basename and by basename without extension. A project Factory created identifies images by an API URL (`.../items/<item id>/image`) whose tail is a fixed word, so the item id is matched too -- name the images after their item id and they resolve. Two tasks resolving to one image is an error, as is an image whose size is not the one the task was annotated on.
- `rectanglelabels`, `polygonlabels`, `brushlabels` and `keypointlabels` convert; anything else is skipped with a warning. Brush masks are the only kind that needs `label_studio_sdk` installed.
