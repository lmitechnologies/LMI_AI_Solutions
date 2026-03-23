import enum
import json
import logging
import os
from dataclasses import asdict, dataclass
from typing import List, Optional, Union

import cv2
import numpy as np
import torch
from pycocotools import mask as coco_mask
from shapely.geometry import Polygon as ShapelyPolygon
from torchvision.ops import masks_to_boxes

from lmi_utils.dataset_utils.mask_encoder import mask2rle, rle2mask
from lmi_utils.gadget_utils.pipeline_utils import fit_array_to_size
from lmi_utils.image_utils.img_resize import resize
from lmi_utils.label_utils.bbox_utils import get_rotated_bbox, rotate

logger = logging.getLogger(__name__)


def _validate_resize_dims(orig_h: int, orig_w: int, new_h: int, new_w: int):
    if orig_w <= 0 or orig_h <= 0:
        raise ValueError("Original dimensions must be positive")
    if new_w <= 0 or new_h <= 0:
        raise ValueError("New dimensions must be positive")


def _validate_flip_dims(flipx: bool, flipy: bool, h: int, w: int):
    if flipx and w <= 0:
        raise ValueError("Width must be positive for horizontal flip")
    if flipy and h <= 0:
        raise ValueError("Height must be positive for vertical flip")


def _require_hw(kwargs):
    h = kwargs.get("h")
    w = kwargs.get("w")
    if not h or not w or h < 0 or w < 0:
        raise ValueError("Height and width are required and must be positive")
    return h, w


class AnnotationType(enum.Enum):
    BOX = "Box"
    POLYGON = "Polygon"
    MASK = "Bitmask"  # This value represents a bitmask annotation.
    KEYPOINT = "Keypoint"


class Base:
    def to_dict(self) -> dict:
        """Convert the dataclass to a dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert the dataclass to a JSON string."""
        return json.dumps(self.to_dict(), indent=4, default=self._default_serializer)

    def save(self, path: str):
        """Save the dataclass as a JSON file."""
        with open(path, "w") as f:
            f.write(self.to_json())

    def _default_serializer(self, obj):
        """Default serializer for non-serializable objects."""
        if isinstance(obj, enum.Enum):
            return obj.value
        if isinstance(obj, Base):
            return obj.to_dict()
        raise TypeError(f"Type {type(obj)} not serializable")

    @classmethod
    def load(cls, path: str) -> "Base":
        """Load a dataclass instance from a JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        if hasattr(cls, "from_dict"):
            return cls.from_dict(data)
        else:
            return cls(**data)


@dataclass
class Point2d(Base):
    x: float
    y: float

    def __post_init__(self):
        self.x = float(self.x)
        self.y = float(self.y)

    @classmethod
    def from_dict(cls, data: dict) -> "Point2d":
        return cls(x=data["x"], y=data["y"])

    def resize(self, orig_h: int, orig_w: int, new_h: int, new_w: int):
        _validate_resize_dims(orig_h, orig_w, new_h, new_w)
        rx = new_w / orig_w
        ry = new_h / orig_h
        self.x *= rx
        self.y *= ry
        return self

    def pad(self, **kwargs):
        pl = kwargs.get("pl", 0)
        pt = kwargs.get("pt", 0)
        self.x += pl
        self.y += pt
        return self

    def flip(self, **kwargs):
        flipx = kwargs.get("flipx", False)
        flipy = kwargs.get("flipy", False)
        h = kwargs.get("h", 0)
        w = kwargs.get("w", 0)
        _validate_flip_dims(flipx, flipy, h, w)
        if flipx:
            self.x = w - self.x
        if flipy:
            self.y = h - self.y
        return self

    def to_numpy(self):
        return np.array([self.x, self.y])

    def coords(self, **kwargs):
        return self.x, self.y

    def to_yolo(self, h, w, **kwargs):
        return [[self.x / w, self.y / h]]


@dataclass
class Box(Base):
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    angle: Optional[float] = 0

    def __post_init__(self):
        self.x_min = float(self.x_min)
        self.y_min = float(self.y_min)
        self.x_max = float(self.x_max)
        self.y_max = float(self.y_max)
        self.angle = float(self.angle)
        if self.x_min > self.x_max:
            raise ValueError("x_min must be less than x_max")
        if self.y_min > self.y_max:
            raise ValueError("y_min must be less than y_max")

    @classmethod
    def from_dict(cls, data: dict) -> "Box":
        return cls(**data)

    def resize(self, orig_h: int, orig_w: int, new_h: int, new_w: int):
        _validate_resize_dims(orig_h, orig_w, new_h, new_w)
        rx = new_w / orig_w
        ry = new_h / orig_h
        self.x_min *= rx
        self.x_max *= rx
        self.y_min *= ry
        self.y_max *= ry
        return self

    def pad(self, **kwargs):
        pl = kwargs.get("pl", 0)
        pt = kwargs.get("pt", 0)
        self.x_min += pl
        self.y_min += pt
        self.x_max += pl
        self.y_max += pt
        return self

    def _rotated_corners(self, **kwargs) -> np.ndarray:
        rot_center = kwargs.get("rot_center", "up_left")
        angle_unit = kwargs.get("angle_unit", "degree")
        return rotate(
            x=self.x_min,
            y=self.y_min,
            w=self.x_max - self.x_min,
            h=self.y_max - self.y_min,
            angle=self.angle,
            rot_center=rot_center,
            unit=angle_unit,
        )

    def flip(self, **kwargs):
        flipx = kwargs.get("flipx", False)
        flipy = kwargs.get("flipy", False)
        h0 = kwargs.get("h", 0)
        w0 = kwargs.get("w", 0)
        _validate_flip_dims(flipx, flipy, h0, w0)

        if self.angle != 0:
            # Get corner points of rotated box
            pts = self._rotated_corners(**kwargs)

            # Flip points
            if flipx:
                pts[:, 0] = w0 - pts[:, 0]
            if flipy:
                pts[:, 1] = h0 - pts[:, 1]

            # Get new rotated bbox and convert to float
            x, y, w, h, angle = get_rotated_bbox(pts)
            x, y, w, h, angle = map(float, (x, y, w, h, angle))

            self.x_min, self.y_min = x, y
            self.x_max, self.y_max = x + w, y + h
            self.angle = angle
        else:
            if flipx:
                self.x_min, self.x_max = w0 - self.x_max, w0 - self.x_min
            if flipy:
                self.y_min, self.y_max = h0 - self.y_max, h0 - self.y_min

        return self

    def to_numpy(self):
        return np.array([self.x_min, self.y_min, self.x_max, self.y_max, self.angle])

    def coords(self, **kwargs):
        return self.x_min, self.y_min, self.x_max, self.y_max, self.angle

    def to_xywh(self):
        """Convert to (x, y, width, height) format."""
        width = self.x_max - self.x_min
        height = self.y_max - self.y_min
        return np.array([self.x_min, self.y_min, width, height, self.angle])

    def area(self):
        return (self.x_max - self.x_min) * (self.y_max - self.y_min)

    def to_coco(self, **kwargs):
        """Convert to COCO format (x_min, y_min, width, height)."""
        return self.to_xywh().tolist()[:4]  # Exclude angle for COCO format

    def to_yolo(self, h, w, **kwargs):
        use_obb = kwargs.get("use_obb", False)

        cx = (self.x_min + self.x_max) / 2
        cy = (self.y_min + self.y_max) / 2
        if self.angle > 0 and use_obb:
            rotated_coords = self._rotated_corners(**kwargs)
            for p in rotated_coords:
                if p[0] > w:
                    raise ValueError(f"Rotated point x value {p[0]} is greater than image width {w}")
                if p[1] > h:
                    raise ValueError(f"Rotated point y value {p[1]} is greater than image height {h}")
            return [[pt[0] / w, pt[1] / h] for pt in rotated_coords]
        else:
            if use_obb:
                logger.debug(f"Use_obb is True but angle is {self.angle}; returning obb formatted bounding box.")
                corners = np.array(
                    [
                        [self.x_min, self.y_min],
                        [self.x_max, self.y_min],
                        [self.x_max, self.y_max],
                        [self.x_min, self.y_max],
                    ]
                )
                return [[pt[0] / w, pt[1] / h] for pt in corners]
            else:
                # convert to center_x, center_y, width, height
                return [
                    [
                        cx / w,
                        cy / h,
                        (self.x_max - self.x_min) / w,
                        (self.y_max - self.y_min) / h,
                    ]
                ]

    def to_mask(self, **kwargs):
        h, w = _require_hw(kwargs)
        mask = np.zeros((h, w), dtype=np.uint8)
        if self.angle != 0:
            pts = self._rotated_corners(**kwargs)
            cv2.fillPoly(mask, [pts], 1)
        else:
            mask[int(self.y_min) : int(self.y_max), int(self.x_min) : int(self.x_max)] = 1
        return Mask(mask=mask)

    def to_polygon(self, **kwargs):
        if self.angle != 0:
            pts = self._rotated_corners(**kwargs)
        else:
            pts = [
                [self.x_min, self.y_min],
                [self.x_max, self.y_min],
                [self.x_max, self.y_max],
                [self.x_min, self.y_max],
            ]
        return Polygon(points=pts)

    def point_in_box(self, x: int, y: int):
        return self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max


@dataclass
class Polygon(Base):
    points: Union[List[List[int]], List[List[float]], np.ndarray] = None

    def __post_init__(self):
        if self.points is None:
            self.points = []
        elif isinstance(self.points, np.ndarray):
            self.points = self.points.astype(float).tolist()

    @classmethod
    def from_dict(cls, data: dict) -> "Polygon":
        return cls(**data)

    def resize(self, orig_h: int, orig_w: int, new_h: int, new_w: int):
        _validate_resize_dims(orig_h, orig_w, new_h, new_w)
        rx = new_w / orig_w
        ry = new_h / orig_h
        for point in self.points:
            point[0] *= rx
            point[1] *= ry
        return self

    def pad(self, **kwargs):
        pl = kwargs.get("pl", 0)
        pt = kwargs.get("pt", 0)
        for point in self.points:
            point[0] += pl
            point[1] += pt
        return self

    def flip(self, **kwargs):
        flipx = kwargs.get("flipx", False)
        flipy = kwargs.get("flipy", False)
        h = kwargs.get("h", 0)
        w = kwargs.get("w", 0)
        _validate_flip_dims(flipx, flipy, h, w)
        for point in self.points:
            if flipx:
                point[0] = w - point[0]
            if flipy:
                point[1] = h - point[1]
        return self

    def to_numpy(self):
        return np.array(self.points)

    def area(self):
        x, y = self.coords()
        return ShapelyPolygon([(int(xi), int(yi)) for xi, yi in zip(x, y)]).area

    def coords(self, **kwargs):
        points = np.array(self.points)
        return points[:, 0].tolist(), points[:, 1].tolist()

    def to_coco(self):
        """convert to COCO format."""
        return np.array(self.points).ravel().tolist()

    def to_yolo(self, h, w, **kwargs):
        return [[point[0] / w, point[1] / h] for point in self.points]

    def to_mask(self, **kwargs):
        h, w = _require_hw(kwargs)
        mask = np.zeros((h, w), dtype=np.uint8)
        pts = self.to_numpy().astype(np.int32)
        cv2.fillPoly(mask, [pts], 1)
        return Mask(mask=mask)

    def to_box(self, **kwargs):
        poly = self.to_numpy()
        x_min = np.min(poly[:, 0])
        y_min = np.min(poly[:, 1])
        x_max = np.max(poly[:, 0])
        y_max = np.max(poly[:, 1])
        return Box(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)

    def to_rbox(self, **kwargs):
        x1, y1, w, h, angle = get_rotated_bbox(self.to_numpy().astype(int))
        return Box(x_min=x1, y_min=y1, x_max=x1 + w, y_max=y1 + h, angle=angle)


@dataclass
class Mask(Base):
    mask: Union[str, np.ndarray]

    def __post_init__(self):
        if not isinstance(self.mask, (str, np.ndarray)):
            raise ValueError("Mask must be a string or numpy array")
        if isinstance(self.mask, np.ndarray):
            self.mask = mask2rle(self.mask)

    @classmethod
    def from_dict(cls, data: dict) -> "Mask":
        return cls(**data)

    def resize(self, orig_h: int, orig_w: int, new_h: int, new_w: int):
        _validate_resize_dims(orig_h, orig_w, new_h, new_w)
        mask_array = rle2mask(self.mask, h=orig_h, w=orig_w)
        resized_mask = resize(mask_array, width=new_w, height=new_h)
        self.mask = mask2rle(resized_mask)
        return self

    def pad(self, **kwargs):
        h, w = _require_hw(kwargs)
        pad_h = kwargs.get("pad_h", 0)
        pad_w = kwargs.get("pad_w", 0)
        mask_array = rle2mask(self.mask, h=h, w=w)
        mask_array, _, _, _, _ = fit_array_to_size(mask_array, pad_w, pad_h)
        self.mask = mask2rle(mask_array)
        return self

    def flip(self, **kwargs):
        flipx = kwargs.get("flipx", False)
        flipy = kwargs.get("flipy", False)
        h, w = _require_hw(kwargs)
        mask_array = rle2mask(self.mask, h=h, w=w)
        if flipx:
            mask_array = np.flip(mask_array, axis=1)
        if flipy:
            mask_array = np.flip(mask_array, axis=0)
        self.mask = mask2rle(mask_array)
        return self

    def to_numpy(self, **kwargs):
        h, w = _require_hw(kwargs)
        return rle2mask(self.mask, h, w)

    def coords(self, **kwargs):
        h, w = _require_hw(kwargs)
        mask = self.to_numpy(h=h, w=w)
        ys, xs = np.nonzero(mask == 1)
        return xs.tolist(), ys.tolist()

    def to_polygon(self, **kwargs) -> List[Polygon]:
        h, w = _require_hw(kwargs)
        mask_array = self.to_numpy(h=h, w=w)
        contours, _ = cv2.findContours(mask_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polygons = [contour.reshape(-1, 2) for contour in contours]
        return [Polygon([[x, y] for x, y in polygon]) for polygon in polygons]

    def to_coco(self, **kwargs):
        """Convert the mask to COCO format."""
        h, w = _require_hw(kwargs)
        mask_array = self.to_numpy(h=h, w=w)
        mask = coco_mask.encode(np.asfortranarray(mask_array.astype(np.uint8)))
        mask["counts"] = mask["counts"].decode("utf-8")
        mask["size"] = [int(dim) for dim in mask["size"]]
        return mask

    def to_yolo(self, h, w, **kwargs):
        instances = []
        for polygon in self.to_polygon(h=h, w=w):
            instances.append(polygon.to_yolo(h, w, **kwargs))
        return instances

    def area(self, **kwargs):
        polygons = self.to_polygon(**kwargs)
        area = 0
        for polygon in polygons:
            area += polygon.area()
        return area

    def to_box(self, **kwargs):
        h, w = _require_hw(kwargs)
        merge_boxes = kwargs.get("merge_boxes", False)
        mask_array = self.to_numpy(h=h, w=w)
        boxes = masks_to_boxes(torch.from_numpy(mask_array).unsqueeze(0))
        if merge_boxes:
            if boxes is not None:
                x_min = boxes[:, 0].min().item()
                y_min = boxes[:, 1].min().item()
                x_max = boxes[:, 2].max().item()
                y_max = boxes[:, 3].max().item()
            else:
                raise ValueError("No boxes found in the mask for merging.")
            return Box(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max, angle=0)
        else:
            if boxes is None or boxes.numel() == 0:
                raise ValueError("No boxes found in the mask.")
            bboxes = []
            for box in boxes:
                x_min, y_min, x_max, y_max = box.tolist()
                bboxes.append(Box(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max, angle=0))
            return bboxes if len(bboxes) > 1 else bboxes[0]  # Return a list if multiple boxes, otherwise a single box


@dataclass
class Label(Base):
    id: str
    color: Optional[str] = None
    annotation_type: AnnotationType = None

    @classmethod
    def from_dict(cls, data: dict) -> "Label":
        return cls(
            id=data["id"],
            color=data.get("color"),
            annotation_type=data.get("annotation_type"),
        )


@dataclass
class Annotation(Base):
    id: str
    label_id: str
    value: Union[Box, Mask, Point2d, Polygon] = None
    type: AnnotationType = None
    link: Optional[str] = None
    confidence: Optional[float] = None
    iou: Optional[float] = None

    def __post_init__(self):
        self.id = str(self.id)
        self.label_id = str(self.label_id)
        self.confidence = float(self.confidence) if self.confidence is not None else None

    @classmethod
    def from_dict(cls, data: dict) -> "Annotation":
        _TYPE_MAP = {
            AnnotationType.BOX.value: BoxAnnotation,
            AnnotationType.MASK.value: MaskAnnotation,
            AnnotationType.KEYPOINT.value: KeypointAnnotation,
            AnnotationType.POLYGON.value: PolygonAnnotation,
        }
        ann_type = data.get("type")
        ann_cls = _TYPE_MAP.get(ann_type)
        if ann_cls is None:
            raise ValueError(f"Unsupported annotation type: {ann_type}")
        return ann_cls.from_dict(data)

    @classmethod
    def _base_fields(cls, data: dict) -> dict:
        """Extract the common Annotation fields from a dict for use in child from_dict methods."""
        return {
            "id": data["id"],
            "label_id": data["label_id"],
            "link": data.get("link"),
            "confidence": data.get("confidence"),
            "iou": data.get("iou"),
        }

    def to_yolo(self, h, w, **kwargs):
        return self.value.to_yolo(h, w, **kwargs)


@dataclass
class BoxAnnotation(Annotation):
    def __post_init__(self):
        self.type = AnnotationType.BOX
        super().__post_init__()

    @classmethod
    def from_dict(cls, data: dict) -> "BoxAnnotation":
        return cls(**cls._base_fields(data), value=Box.from_dict(data["value"]))


@dataclass
class MaskAnnotation(Annotation):
    def __post_init__(self):
        self.type = AnnotationType.MASK
        super().__post_init__()

    @classmethod
    def from_dict(cls, data: dict) -> "MaskAnnotation":
        return cls(**cls._base_fields(data), value=Mask.from_dict(data["value"]))


@dataclass
class KeypointAnnotation(Annotation):
    bounding_box_id: Optional[str] = None

    def __post_init__(self):
        self.type = AnnotationType.KEYPOINT
        super().__post_init__()

    @classmethod
    def from_dict(cls, data: dict) -> "KeypointAnnotation":
        return cls(**cls._base_fields(data), value=Point2d.from_dict(data["value"]), bounding_box_id=data.get("bounding_box_id"))


@dataclass
class PolygonAnnotation(Annotation):
    def __post_init__(self):
        self.type = AnnotationType.POLYGON
        super().__post_init__()

    @classmethod
    def from_dict(cls, data: dict) -> "PolygonAnnotation":
        return cls(**cls._base_fields(data), value=Polygon.from_dict(data["value"]))


@dataclass
class FileAnnotations(Base):
    id: str  # File ID
    path: str  # File path
    height: int  # File height
    width: int  # File width
    annotations: List[Annotation] = None
    predictions: List[Annotation] = None

    def __post_init__(self):
        self.annotations = self.annotations or []
        self.predictions = self.predictions or []

    @classmethod
    def from_dict(cls, data: dict) -> "FileAnnotations":
        annotations = [Annotation.from_dict(a) for a in data.get("annotations", [])]
        predictions = [Annotation.from_dict(a) for a in data.get("predictions", [])]
        return cls(
            id=data["id"],
            path=data["path"],
            height=data.get("height", None),
            width=data.get("width", None),
            annotations=annotations,
            predictions=predictions,
        )

    @property
    def has_annotations(self) -> bool:
        return len(self.annotations) > 0

    def relative_path(self, base_path: str) -> str:
        return os.path.relpath(self.path, base_path)

    def update_file(self, id, path, height, width):
        self.id = id
        self.path = path
        self.height = height
        self.width = width
        return self

    def _get_target_list(self, list_type: str) -> List[Annotation]:
        if list_type not in ("annotations", "predictions"):
            raise ValueError("list_type must be either 'annotations' or 'predictions'")
        return self.annotations if list_type == "annotations" else self.predictions

    def delete_annotation(self, annotation_id: str, list_type: str = "annotations") -> bool:
        target_list = self._get_target_list(list_type)
        for index, ann in enumerate(target_list):
            if ann.id == annotation_id:
                del target_list[index]
                logger.debug(f"Deleted annotation with id '{annotation_id}' from {list_type}.")
                return True
        logger.warning(f"Annotation with id '{annotation_id}' not found in {list_type}.")
        return False

    def get_annotations_by_type(self, annotation_type: AnnotationType, list_type: str = "annotations") -> List[Annotation]:
        return [ann for ann in self._get_target_list(list_type) if ann.type == annotation_type]

    def update_annotations(self, annotations: List[Annotation], list_type: str = "annotations"):
        self._get_target_list(list_type)  # validates list_type
        setattr(self, list_type, annotations)

    def assign_keypoints(self, target_ids=None):
        target_ids = target_ids or []
        for annotation in self.annotations:
            if annotation.type == AnnotationType.KEYPOINT:
                assigned = False
                for box in self.annotations:
                    if len(target_ids) > 0 and box.label_id not in target_ids:
                        continue
                    if box.type == AnnotationType.BOX and box.value.point_in_box(annotation.value.x, annotation.value.y):
                        if annotation.bounding_box_id is None:
                            annotation.bounding_box_id = box.id
                            assigned = True
                            break

                if not assigned:
                    raise Exception(f"Keypoint {annotation.id} not assigned to any box")

        return self

    def to_yolo(
        self,
        label_id_idx: dict,
        to_segmentation=False,
        to_object_detection=False,
        merge_boxes=False,
        target_classes=None,
        use_obb=False,
    ):
        """Convert this file's annotations to YOLO format.
        `label_to_index` is a function mapping a label id to an integer index.
        """
        target_classes = target_classes or []
        yolo_annotations = []
        h = self.height
        w = self.width
        yolo_annotations_map = {}
        label_ids = []

        for annotation in self.annotations:
            if annotation.type == AnnotationType.KEYPOINT:
                continue
            updated_annotations = []
            if len(target_classes) > 0 and annotation.label_id not in target_classes:
                continue

            # Conversion steps:
            if annotation.type == AnnotationType.BOX and to_segmentation:
                logger.debug(f"Converting box {annotation.id} to YOLO format with mask_type=AnnotationType.MASK")
                updated_annotations.append(annotation.value.to_mask(h=h, w=w))
            elif annotation.type == AnnotationType.MASK and to_object_detection:
                logger.debug(f"Converting mask {annotation.id} to YOLO format with mask_type=AnnotationType.MASK")
                updated_annotations.append(annotation.value.to_box(h=h, w=w, merge_boxes=merge_boxes))
            elif annotation.type == AnnotationType.POLYGON and to_object_detection:
                logger.debug(f"Converting polygon {annotation.id} to YOLO format with mask_type=AnnotationType.POLYGON")
                updated_annotations.append(annotation.value.to_box(h=h, w=w))

            converted = (
                [ann.to_yolo(h, w, use_obb=use_obb) for ann in updated_annotations]
                if updated_annotations
                else [annotation.to_yolo(h, w, use_obb=use_obb)]
            )
            for conv in converted:
                if annotation.type == AnnotationType.MASK:
                    instance = []
                    for p in conv:
                        instance = [label_id_idx[annotation.label_id]] + np.array(p).flatten().tolist()
                        yolo_annotations.append(instance)
                else:
                    instance = [label_id_idx[annotation.label_id]] + np.array(conv).flatten().tolist()
                    yolo_annotations.append(instance)
                yolo_annotations_map[annotation.id] = instance
            label_ids.append(annotation.label_id)

        # handle converting keypoints to YOLO format
        # assign keypoints to bounding boxes

        self.assign_keypoints(target_ids=target_classes)
        for annotation in self.annotations:
            if annotation.type == AnnotationType.KEYPOINT:
                logger.debug(f"Converting keypoint {annotation.id} to YOLO format with bounding box {annotation.bounding_box_id}")
                box = yolo_annotations_map.get(annotation.bounding_box_id)

                if box is None:
                    raise Exception(f"Bounding box {annotation.bounding_box_id} not found for keypoint {annotation.id}")
                yolo_kp = annotation.to_yolo(h, w, use_obb=use_obb)
                # Extend the box annotation in-place (list reference is shared with yolo_annotations)
                box.extend(np.array(yolo_kp).flatten().tolist())

        if len(yolo_annotations) == 0:
            logger.debug(f"No annotations found for file {self.path}")
        return yolo_annotations, label_ids


@dataclass
class Dataset(Base):
    labels: List[Label]
    files: List[FileAnnotations]

    @classmethod
    def from_dict(cls, data: dict) -> "Dataset":
        labels = [Label.from_dict(li) for li in data.get("labels", [])]
        files = [FileAnnotations.from_dict(f) for f in data.get("files", [])]
        return cls(labels=labels, files=files)

    @classmethod
    def load(cls, file_path: str) -> "Dataset":
        with open(file_path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)

    @property
    def base_path(self) -> str:
        all_files = [file_ann.path for file_ann in self.files]
        common_prefix = os.path.commonprefix(all_files)
        return os.path.dirname(common_prefix)

    def delete_empty_files(self):
        """Delete files that have no annotations."""
        self.files = [file_ann for file_ann in self.files if file_ann.has_annotations]
        return self

    def files_to_relative(self):
        base_path = self.base_path
        if os.path.isabs(base_path):
            for file_ann in self.files:
                file_ann.path = file_ann.relative_path(base_path)
        return self

    def get_label_ids(self) -> List[str]:
        return [label.id for label in self.labels]

    def label_to_index(self, label_id: str) -> int:
        for idx, label in enumerate(self.labels):
            if label.id == label_id:
                return idx
        raise ValueError(f"Label id {label_id} not found.")

    def delete_label(self, label_id: str):
        self.labels = [label for label in self.labels if label.id != label_id]
        for file_ann in self.files:
            file_ann.annotations = [ann for ann in file_ann.annotations if ann.label_id != label_id]
            file_ann.predictions = [ann for ann in file_ann.predictions if ann.label_id != label_id]
        return self

    def to_yolo(self, **kwargs):
        to_segmentation = kwargs.get("to_segmentation", False)
        to_object_detection = kwargs.get("to_object_detection", False)
        merge_boxes = kwargs.get("merge_boxes", False)
        target_classes = kwargs.get("target_classes", ["all"])
        class_map = kwargs.get("class_map", {})
        target_label_ids = []
        if target_classes != ["all"]:
            # delete the annotations that are not in the target classes
            delete_ids = [label.id for label in self.labels if label.id not in target_classes]
            self.labels = [label for label in self.labels if label.id in target_classes]
            for file_ann in self.files:
                file_ann.annotations = [ann for ann in file_ann.annotations if ann.label_id not in delete_ids]
                file_ann.predictions = [ann for ann in file_ann.predictions if ann.label_id not in delete_ids]
            logger.debug(f"Deleted annotations for labels {delete_ids}")

            target_label_ids = target_classes
            logger.debug(f"Updated label ids {self.labels}")
        else:
            target_label_ids = [label.id for label in self.labels]
            logger.debug(f"Using all labels {target_label_ids}")

        # generate label counts
        if class_map:
            if len(class_map) != len(self.labels):
                raise ValueError("Class map must have the same number of classes as the dataset")
            label_id_index = {name: idx for idx, name in class_map.items()}
        else:
            label_id_index = {}
            label_idx = 0
            # generate label id index for labels that have annotations
            # create a sequential index for the labels
            for file_ann in self.files:
                for annotation in file_ann.annotations:
                    if annotation.label_id in target_label_ids and annotation.label_id not in label_id_index:
                        label_id_index[annotation.label_id] = label_idx
                        label_idx += 1

        # sort the label_id_index by label id
        label_id_index = dict(sorted(label_id_index.items(), key=lambda item: item[1]))

        n_kpts = 0
        image_to_labels = {}
        label_ids = []
        for file_ann in self.files:
            file_path = file_ann.path

            logger.debug(f"Processing file {file_path}")
            if file_path not in image_to_labels:
                image_to_labels[file_path] = []

            keypoints = file_ann.get_annotations_by_type(AnnotationType.KEYPOINT)
            if keypoints:
                if n_kpts == 0:
                    n_kpts = len(keypoints)
                elif len(keypoints) != n_kpts:
                    raise Exception(f"Inconsistent number of keypoints: expected {n_kpts}, found {len(keypoints)}")

            # Call the file-level to_yolo method:
            file_yolo, file_label_ids = file_ann.to_yolo(
                label_id_idx=label_id_index,
                to_segmentation=to_segmentation,
                to_object_detection=to_object_detection,
                merge_boxes=merge_boxes,
                target_classes=target_label_ids,
                use_obb=kwargs.get("use_obb", False),
            )
            label_ids.extend(file_label_ids)
            image_to_labels[file_path].extend(file_yolo)

        # generate the class map
        return dict(
            image_labels=image_to_labels,
            class_map=label_id_index,
            n_kpts=n_kpts,
        )
