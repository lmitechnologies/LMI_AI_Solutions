import logging

import numpy as np

# LMI packages
from lmi_utils.dataset_utils.representations import (
    Annotation,
    AnnotationType,
    Box,
    Point2d,
    PolygonAnnotation,
)
from lmi_utils.gadget_utils.pipeline_utils import fit_array_to_size

logger = logging.getLogger(__name__)


def fit_shapes_to_size(shapes, pad_l, pad_t, pad_h, pad_w, orig_h, orig_w):
    """
    description:
        add the left and top paddings to the shapes
    arguments:
        shapes(list): a list of Shape objects (Rect or Mask)
        pad_l(int): the left paddings
        pad_t(int): the top paddings
    """
    shapes = shapes.copy()
    for annot in shapes:
        annot.value = annot.value.pad(pad_h=pad_h, pad_w=pad_w, pl=pad_l, pt=pad_t, h=orig_h, w=orig_w)
    return shapes


def pad_annotated_image(
    image: np.ndarray, annotations: list[Annotation], width: int, height: int, value: int = 0
) -> tuple[np.ndarray, list[Annotation], bool]:
    """
    description:
        pad the image to the size [width,height] and modify its annotations accordingly
    arguments:
        image(np.ndarray): the input image
        annotations(list): a list of annotation objects
        width(int): the target width of the output image
        height(int): the target height of the output image
        value(int): the image pad fill value
    return:
        im_out(np.ndarray): the padded image
        annotations(list): a list of annotation objects
        is_warning(bool): whether the annotations have been clipped or removed
    """
    h, w = image.shape[:2]
    pw = width
    ph = height

    if ph is None and pw is None:
        ph = h
        pw = w

    # pad image
    im_out, pad_l, _, pad_t, _ = fit_array_to_size(image, pw, ph, value=value)
    pw = im_out.shape[1]
    ph = im_out.shape[0]

    # pad shapes
    annotations = fit_shapes_to_size(annotations, pad_l, pad_t, pad_h=ph, pad_w=pw, orig_h=h, orig_w=w)

    annotations, is_warning = clip_shapes(annotations, W=pw, H=ph)

    return im_out, annotations, is_warning


def pad_dataset(dataset, images, output_imsize, crop_warning_level=logging.DEBUG, value=0):
    """
    pad/crop the image to the size [W,H] and modify its annotations accordingly
    arguments:
        input_path(str): the input image path
        json_path(str): the path to the json annotation file
        output_imsize(list): the width and height of the output image
        value(int): the image pad fill value
    """

    padded_images = {}
    cnt_warnings = 0
    for f in dataset.files:
        file_path = f.path

        im_out, annot_out, is_warning = pad_annotated_image(
            image=images[file_path],
            annotations=f.annotations,
            width=output_imsize[0],
            height=output_imsize[1],
            value=value,
        )

        if is_warning:
            cnt_warnings += 1

        f.height = im_out.shape[0]
        f.width = im_out.shape[1]
        f.annotations = annot_out
        padded_images[file_path] = im_out
    if cnt_warnings:
        logger.log(crop_warning_level, f"Labels were cropped for {cnt_warnings} images.")
    return padded_images, dataset


def clip_shapes(shapes, W, H, crop_warning_level=logging.DEBUG):
    """Clip the shapes to the target size [W,H].

    Returns:
        tuple: the shapes inside, one per piece of a polygon the edge cut apart; and whether any shape was changed or removed
    """
    is_warning = False
    delete_ids = []
    pieces = {}
    for shape in shapes:
        is_del = 0
        if shape.type == AnnotationType.BOX and shape.value.angle != 0:
            # clip the rotated footprint, not the unrotated bounds, and refit a rotated box to what is left
            corners = shape.value.to_polygon()
            inside = corners.clip(W, H)
            if not inside:
                is_del = 1
                delete_ids.append(shape.id)
                logger.log(crop_warning_level, f"Rotated box {shape.id} was excluded by image dimensions [{W},{H}]")
            elif inside[0] is not corners:
                logger.log(crop_warning_level, f"Rotated box {shape.id} was clipped to image dimensions [{W}, {H}]")
                is_warning = True
                shape.value = inside[0].to_rbox()

        elif shape.type == AnnotationType.BOX:
            box = shape.value.to_numpy()[:-1]
            new_box = np.clip(box, a_min=0, a_max=[W, H, W, H])

            if np.all(new_box == 0) or new_box[0] == new_box[2] or new_box[1] == new_box[3]:
                is_del = 1
                delete_ids.append(shape.id)
                logger.log(
                    crop_warning_level,
                    f"Bounding box {box} was excluded by image dimensions [{W},{H}]",
                )
            elif (
                (np.any(new_box == W) and np.all(box != W))
                or (np.any(new_box == H) and np.all(box != H))
                or (np.any(new_box == 0) and np.all(box != 0))
            ):
                logger.log(
                    crop_warning_level,
                    f"Bounding box {box} was clipped to image dimensions [{W}, {H}]",
                )
                is_warning = True

                shape.value = Box(*new_box)

        elif shape.type == AnnotationType.MASK:
            # Mask.pad already cropped the mask to [W,H]
            if not shape.value.to_numpy(h=H, w=W).any():
                is_del = 1
                delete_ids.append(shape.id)
                logger.log(crop_warning_level, f"Mask {shape.id} was excluded by image dimensions [{W},{H}]")

        elif shape.type == AnnotationType.POLYGON:
            inside = shape.value.clip(W, H)
            if not inside:
                is_del = 1
                delete_ids.append(shape.id)
                logger.log(crop_warning_level, f"Polygon {shape.id} was excluded by image dimensions [{W},{H}]")
            elif inside[0] is not shape.value:
                logger.log(crop_warning_level, f"Polygon {shape.id} was clipped to image dimensions [{W}, {H}]")
                is_warning = True
                if len(inside) == 1:
                    shape.value = inside[0]
                else:
                    # each piece of an outline the edge cut apart is its own label, named as dataset_tile names them
                    pieces[shape.id] = [
                        PolygonAnnotation(
                            f"{shape.id}_p{i}", shape.label_id, part, link=shape.link, confidence=shape.confidence, iou=shape.iou
                        )
                        for i, part in enumerate(inside)
                    ]

        elif shape.type == AnnotationType.KEYPOINT:
            x, y = shape.value.coords()
            if x < 0 or x > W or y < 0 or y > H:
                is_del = 1
                logger.log(
                    crop_warning_level,
                    f"keypoint ({x},{y}) was excluded by image dimensions [{W},{H}]",
                )
                delete_ids.append(shape.id)
            else:
                shape.value = Point2d(x=x, y=y, visibility=shape.value.visibility)

        if is_del:
            is_warning = True

    # Keep linked pose annotations atomic when cropping removes their owning box.
    deleted_box_ids = {shape.id for shape in shapes if shape.type == AnnotationType.BOX and shape.id in delete_ids}
    delete_ids.extend(
        shape.id
        for shape in shapes
        if shape.type == AnnotationType.KEYPOINT
        and getattr(shape, "bounding_box_id", None) in deleted_box_ids
        and shape.id not in delete_ids
    )

    kept = []
    for shape in shapes:
        if shape.id not in delete_ids:
            kept.extend(pieces.get(shape.id, [shape]))
    return kept, is_warning
