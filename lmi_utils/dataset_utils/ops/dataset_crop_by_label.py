import logging

from lmi_utils.dataset_utils.ops.dataset_pad import clip_shapes
from lmi_utils.dataset_utils.representations import AnnotationType, Mask

logger = logging.getLogger(__name__)


def crop_dataset_by_label(dataset, images, target_label, crop_warning_level=logging.DEBUG):
    """Crop each image to its one target_label box, and cut its other labels at the crop's edge as padding does.

    The crop covers the pixels whose centers are inside the box. Files without the label lose their annotations.
    """
    crop_boxes = {}
    for file in dataset.files:
        targets = [annot for annot in file.annotations if annot.label_id == target_label]
        if not targets:
            logger.warning(f"no label found in {file.path}")
            file.annotations = []
            continue
        if len(targets) > 1:
            raise ValueError(f"more than one label found in {file.path}")
        if targets[0].type != AnnotationType.BOX:
            raise ValueError(f"the {target_label} label in {file.path} is not a box")
        if targets[0].value.angle:
            raise ValueError(f"the {target_label} box in {file.path} is rotated; rotated crops are not supported")
        crop_boxes[file.path] = targets[0]

    dataset.delete_empty_files()
    dataset.delete_label(target_label)

    cropped_images = {}
    cnt_warnings = 0
    for file in dataset.files:
        target = crop_boxes[file.path]
        image = images[file.path]
        if image is None:
            raise ValueError(f"failed to read {file.path}")
        h, w = image.shape[:2]
        x1, y1, x2, y2 = target.value.pixel_bounds()
        x1, x2 = min(w, x1), min(w, x2)
        y1, y2 = min(h, y1), min(h, y2)
        cw, ch = x2 - x1, y2 - y1
        if not cw or not ch:
            raise ValueError(f"the {target_label} box in {file.path} covers no pixel")

        # the crop box's keypoints go with it, as padding drops the keypoints of a box it removes
        file.annotations = [a for a in file.annotations if getattr(a, "bounding_box_id", None) != target.id]
        for annot in file.annotations:
            if annot.type == AnnotationType.MASK:
                annot.value = Mask(mask=annot.value.to_numpy(h=h, w=w)[y1:y2, x1:x2].copy())
            else:
                annot.value = annot.value.pad(pl=-x1, pt=-y1)
        file.annotations, is_warning = clip_shapes(file.annotations, W=cw, H=ch, crop_warning_level=crop_warning_level)
        cnt_warnings += is_warning

        cropped_images[file.path] = image[y1:y2, x1:x2]
        file.height, file.width = ch, cw

    if cnt_warnings:
        logger.log(crop_warning_level, f"Labels were cropped for {cnt_warnings} images.")
    return cropped_images, dataset
