# Converting datasets to Factory

Factory imports one directory format. COCO, YOLO and Label Studio datasets reach it through the AIS dataset json:

```
COCO ─────────coco_to_json──┐
YOLO ─────────yolo_to_json──┼──> labels.json ──json_to_factory──> Factory directory
Label Studio ──lst_to_json──┘
```

The json in the middle keeps the existing tools (`json_to_yolo`, `json_to_coco`, `apply_ops`, `plot_with_json`) working on a converted
dataset, and leaves Factory with one writer. `coco_to_json` and `yolo_to_json` write `labels.json` where `json_to_factory` looks for it,
so the second command below never needs `-j`.

## Pose datasets need a declared schema

A keypoint layout is a *declaration*, not a tally of the annotations that happen to be present.
Factory reads it from a `.meta.json` at the dataset root:

```json
{
  "annotationSchema": {
    "type": "Pose",
    "version": 1,
    "coordinateDimensions": 3,
    "keypoints": {
      "head": { "name": "Head" },
      "left-flange": { "name": "Left flange" },
      "right-flange": { "name": "Right flange" }
    },
    "classes": {
      "bolt": {
        "name": "Hex bolt",
        "keypointIds": ["head", "left-flange", "right-flange"],
        "horizontalFlipPairs": [["left-flange", "right-flange"]],
        "skeleton": [[0, 1], [0, 2]]
      }
    }
  }
}
```

| Field | Required | Meaning |
| --- | --- | --- |
| `type`, `version`, `coordinateDimensions` | yes | Always `"Pose"`, `1` and `3`. A 2D source is normalized to 3 on import. |
| `keypoints` | yes | Every keypoint the classes draw slots from, keyed by keypoint id, each with the `name` a person reads. |
| `classes` | yes | Keyed by class id, one entry per box class that owns keypoints, each with a `name`. |
| `keypointIds` | yes | The class's slot order, as keypoint ids; its length is that class's K. A slot no image observes stays declared. |
| `horizontalFlipPairs` | yes, may be `null` | Keypoint ids that trade places under a mirror; anything unpaired is its own mirror. `null` imports fine but blocks `fliplr`/`flipud` in training. `[]` declares a class that mirrors onto itself. Omitting the field is an error — write `null`. |
| `skeleton` | no | Visualization only, zero-based. Omit it when there is none; `null` is an error. |

Keypoint annotations name their slot in `label_id` and their instance in `bounding_box_id`.

## Ids and names are separate

Every class and keypoint has a permanent **id** and a **name** someone can change.
Annotations, class maps and model contracts all store the id, so renaming a class in Factory strands nothing.

Ids are *derived from* names, never allocated: a COCO category `person` and a YOLO class `person`, converted months apart, arrive at the same id without consulting each other.
The converters do this for you — `class_map`, `flip_map`, `kpt_names` and `--keypoint_names` are all written in names, and nothing asks you for an id.
A name already usable as an id is kept as it stands, which is why `bolt` and `left-flange` above look unchanged; anything else becomes a readable stem plus a digest, so `Left Eye` derives `left_eye_4c96d10f1b2a`.

Two rules follow, both enforced when the schema is built:

- **Ids must be valid.** Building a `Label` by hand with `id="hex bolt"` is rejected; derive it with `derive_pose_id` from `lmi_utils.dataset_utils.pose_identifiers`.
- **Names must be unambiguous** within their vocabulary, and may not be another entry's id. Label Studio resolves a region's label by its alias *or* its displayed value, so a repeat is ambiguous to the editor itself. Names also may not contain `$`, which Label Studio reads as a task-data substitution.

Because two classes both declaring `left_eye` mean the same keypoint, `keypoints` is one flat vocabulary for the schema rather than a list per class — which is also the shape Label Studio's own labeling configuration takes.

`derive_pose_id` is a persisted contract shared with GoFactory's TypeScript and Python SDKs; the three copies must agree byte for byte, and `tests/lmi_utils/dataset_utils/test_pose_identifiers.py` holds the frozen vectors that say so.

## Getting a schema

Two ways, neither of which is editing by hand.

**1. Let a converter emit it.** `coco_to_json`, `yolo_to_json` derive what they can from the source, and `json_to_factory` writes the result into every directory the dataset can be imported from. What the source leaves you to supply:

| Source | Supplies | You add |
| --- | --- | --- |
| COCO | `categories[].keypoints` and `[].skeleton` | `--flip_map` — COCO states no mirror symmetry |
| YOLO | `kpt_names`/`kpt_shape` layouts, and mirror pairs from `flip_idx` | `--keypoint_names`, only when the yaml declares no `kpt_names` |

**2. Author one** — for Label Studio, or to give several datasets one shared declaration.
`build_pose_schema` takes the same `Label` objects the converters use and validates the result:

```python
import json
from lmi_utils.dataset_utils.pose_identifiers import derive_pose_id
from lmi_utils.dataset_utils.representations import Label
from lmi_utils.label_utils.json_to_factory import build_pose_schema

schema = build_pose_schema(
    [
        Label(id="bolt", name="Hex bolt", keypoint_ids=["head", "left-flange", "right-flange"],
              horizontal_flip_pairs=[["left-flange", "right-flange"]]),
        # A name that is not usable as an id has to be derived, the same way a converter would
        Label(id=derive_pose_id("Tab strip"), name="Tab strip",
              keypoint_ids=["left-edge", "right-edge"], horizontal_flip_pairs=None),
    ],
    {"head": "Head", "left-flange": "Left flange", "right-flange": "Right flange"},
)
json.dump(schema, open("schema.json", "w"), indent=2)
```

Its output is what `lst_to_json -ps` expects, and the fields carry through `labels.json` on `Label` as `name`, `keypoint_ids`, `horizontal_flip_pairs` and `skeleton`, with the keypoint names on the dataset's own `keypoints` map.

## COCO

```bash
python3 -m lmi_utils.label_utils.coco_to_json -j annotations.json -i images/
python3 -m lmi_utils.label_utils.json_to_factory -i images/ -o factory_dataset/

# Rename classes, declare mirror pairs, read zero-based skeletons, keep segmentation
python3 -m lmi_utils.label_utils.coco_to_json -j annotations.json -i images/ \
  --class_map class_map.json --flip_map flip_map.json --skeleton_base 0 --segmentation
```

- `--class_map` renames a COCO category: `{"Bolt": "Hex bolt"}`. Without it the class keeps the category name. Either way the
  name is what the class shows and what its id is derived from.
- `--flip_map` maps a class name to its mirror pairs, in keypoint names: `{"Hex bolt": [["left-flange", "right-flange"]]}`.
  **COCO declares no flip symmetry**, so without it every class gets `horizontalFlipPairs: null`.
- `--skeleton_base 0` for zero-based skeleton indices; the COCO convention of one-based is assumed.
- `--segmentation` writes classes with no keypoints as polygons or bitmasks instead of boxes.
- A keypoint with visibility 0 leaves its declared slot empty rather than landing at the origin.

## YOLO

```bash
python3 -m lmi_utils.label_utils.yolo_to_json -y dataset.yaml
python3 -m lmi_utils.label_utils.json_to_factory -i <dataset root> -o factory_dataset/

# Name the slots of a yaml that declares no kpt_names, and convert two splits instead of three
python3 -m lmi_utils.label_utils.yolo_to_json -y dataset.yaml \
  --keypoint_names head,left-flange,right-flange --splits train,val
```

- `names` becomes the class names, from which the class ids are derived, and `kpt_shape` the slot count. A yaml with no keypoints is
  unaffected: its class names stay its class ids, exactly as before.
- `kpt_names` names each class's slots, keyed by class index (as Ultralytics writes it) or class name (as Factory's exporter does).
  Slots marked `__unused_*` are dropped. Without it, every class gets all K slots, named by `--keypoint_names` or positionally.
- The global `flip_idx` is restated as each class's own mirror pairs. A class whose mirror leaves the slots it owns declares no symmetry
  rather than a partial one.
- `--splits` defaults to `train,val,test`. Each becomes one dataset, images keeping their path relative to the dataset root so
  same-named images stay distinct.
- A stale `path` in the yaml falls back to the dataset root, or to `--root`.

## Label Studio

Export the full `JSON`, **not** `JSON-MIN`, which drops fields the converter needs.

```bash
# With the schema of the Factory dataset the project came from
python3 -m lmi_utils.label_utils.lst_to_json -i export.json -imgs images/ -of images/labels.json -ps source_dataset/
python3 -m lmi_utils.label_utils.json_to_factory -i images/ -o factory_dataset/

# No schema yet: draft one from the first export, edit it, then pass it to every export
python3 -m lmi_utils.label_utils.lst_to_json -i export1.json -imgs images1/ -of images1/labels.json --scaffold_schema draft.json
# edit draft.json: fix the slot order, add the keypoints this export happens not to show, fill in horizontalFlipPairs or put a null
python3 -m lmi_utils.label_utils.lst_to_json -i export1.json -imgs images1/ -of images1/labels.json -ps draft.json
python3 -m lmi_utils.label_utils.lst_to_json -i export2.json -imgs images2/ -of images2/labels.json -ps draft.json

# Drop keypoints that belong to no box instead of rejecting the dataset
python3 -m lmi_utils.label_utils.json_to_factory -i images/ -o factory_dataset/ --unlinked_keypoints drop
```

`-ps` takes a Factory dataset directory, its `.meta.json`, or a bare schema file. Skip it and the export still converts, but keypoint
labels become classes of their own and the result is not pose-trainable.

A project Factory generated stores each label under its own id, so an export names ids rather than the text the annotator saw, and the
round trip survives a rename. An export from a project built by hand names its labels however the annotator did, and those values are
read as they stand.

- A project Factory created already has a schema, on the annotation project and on its source dataset; the project id is in each task's
  image URL.
- Otherwise `--scaffold_schema` drafts one, which needs finishing before use: it carries only the keypoints its own export happens to
  show, in the order the annotator worked, and always leaves `horizontalFlipPairs: null`.
- **Use one schema for every dataset of the same classes.** Factory will not train on datasets whose shared class declares different
  keypoints or mirror pairs — the orders may differ, the keypoints themselves may not. Drafting per export invites exactly that.

### Other notes

- A keypoint label the schema declares becomes a slot of its class; one naming no declared slot is rejected.
- `--unlinked_keypoints` is `error` (default), `drop` or `keep`, since a pose model cannot train on a keypoint no box owns.
- Keypoints link to their box by `parentID`, falling back to older relation records and then to containment, which is ambiguous when
  boxes overlap.
- Name images after their Factory item id so URL-identified images resolve. Two tasks resolving to one image is an error, as is a size
  that differs from the annotated one.
- Keypoints arrive with no visibility, which Factory reads as visible.
- `rectanglelabels`, `polygonlabels`, `brushlabels` and `keypointlabels` convert; anything else is skipped with a warning.
  Brush masks need `label_studio_sdk`.
