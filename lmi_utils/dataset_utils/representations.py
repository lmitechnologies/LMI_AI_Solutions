import enum
import json
import logging
import os
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Union

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


def _pixel_points(points) -> np.ndarray:
    """Round points to the pixel grid for OpenCV rasterization, which only accepts integer coordinates."""
    return np.round(np.asarray(points, dtype=float)).astype(np.int32)


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
        # create directory if it doesn't exist; a bare filename has none
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
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
    visibility: Optional[int] = None

    def __post_init__(self):
        self.x = float(self.x)
        self.y = float(self.y)

    @classmethod
    def from_dict(cls, data: dict) -> "Point2d":
        return cls(x=data["x"], y=data["y"], visibility=data.get("visibility"))

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

    def _corners(self, **kwargs) -> np.ndarray:
        """The box's four corners, rotated when it carries an angle."""
        if self.angle != 0:
            return self._rotated_corners(**kwargs).astype(float)
        return np.array(
            [
                [self.x_min, self.y_min],
                [self.x_max, self.y_min],
                [self.x_max, self.y_max],
                [self.x_min, self.y_max],
            ],
            dtype=float,
        )

    def _enclosing_bounds(self, **kwargs) -> tuple:
        """The axis-aligned extent enclosing the box, its rotation included.

        A rotated box's own x_min..x_max is the extent it had before rotating, which is not where the
        object is, so any axis-aligned consumer has to measure the rotated corners instead.
        """
        pts = self._corners(**kwargs)
        return pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()

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

    def area(self, **kwargs):
        return (self.x_max - self.x_min) * (self.y_max - self.y_min)

    def to_coco(self, **kwargs):
        """Convert to COCO format (x_min, y_min, width, height); a COCO bbox is axis-aligned."""
        x_min, y_min, x_max, y_max = self._enclosing_bounds(**kwargs)
        return [float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min)]

    def to_yolo(self, h, w, **kwargs):
        if kwargs.get("use_obb", False):
            corners = self._corners(**kwargs)
            # Rotation may move valid edge boxes slightly outside the image; Ultralytics expects clipped normalized corners.
            corners[:, 0] = np.clip(corners[:, 0], 0, w)
            corners[:, 1] = np.clip(corners[:, 1], 0, h)
            return [[x / w, y / h] for x, y in corners]

        # center_x, center_y, width, height of the axis-aligned extent
        x_min, y_min, x_max, y_max = self._enclosing_bounds(**kwargs)
        x_min, x_max = np.clip([x_min, x_max], 0, w)
        y_min, y_max = np.clip([y_min, y_max], 0, h)
        return [[(x_min + x_max) / 2 / w, (y_min + y_max) / 2 / h, (x_max - x_min) / w, (y_max - y_min) / h]]

    def to_mask(self, **kwargs):
        h, w = _require_hw(kwargs)
        mask = np.zeros((h, w), dtype=np.uint8)
        if self.angle != 0:
            cv2.fillPoly(mask, [_pixel_points(self._rotated_corners(**kwargs))], 1)
        else:
            mask[int(self.y_min) : int(self.y_max), int(self.x_min) : int(self.x_max)] = 1
        return Mask(mask=mask)

    def to_polygon(self, **kwargs):
        return Polygon(points=self._corners(**kwargs).tolist())

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

    def area(self, **kwargs):
        x, y = self.coords(**kwargs)
        return ShapelyPolygon([(int(xi), int(yi)) for xi, yi in zip(x, y)]).area

    def coords(self, **kwargs):
        points = np.array(self.points)
        return points[:, 0].tolist(), points[:, 1].tolist()

    def to_coco(self, **kwargs):
        """convert to COCO format."""
        return [np.array(self.points).ravel().tolist()]

    def to_yolo(self, h, w, **kwargs):
        if kwargs.get("use_obb", False):
            # an obb row is exactly four corners, so an arbitrary outline has to become its rotated box
            return self.to_rbox(**kwargs).to_yolo(h, w, **kwargs)
        return [[point[0] / w, point[1] / h] for point in self.points]

    def encloses_area(self, **kwargs) -> bool:
        """Whether the outline encloses at least one pixel; a point, a line, or collinear points do not."""
        return cv2.contourArea(self.to_numpy().astype(np.float32).reshape(-1, 2)) >= 1

    def to_mask(self, **kwargs):
        h, w = _require_hw(kwargs)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [_pixel_points(self.to_numpy())], 1)
        return Mask(mask=mask)

    def to_box(self, **kwargs):
        poly = self.to_numpy()
        x_min = np.min(poly[:, 0])
        y_min = np.min(poly[:, 1])
        x_max = np.max(poly[:, 0])
        y_max = np.max(poly[:, 1])
        return Box(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)

    def to_rbox(self, **kwargs):
        x1, y1, w, h, angle = get_rotated_bbox(self.to_numpy())
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

    def to_polygons(self, **kwargs) -> List[Polygon]:
        """One outline per connected region, skipping any that encloses no area.

        Plural because a mask may hold several disconnected regions; a Box, which is always one connected
        shape, has the singular `to_polygon`.
        """
        h, w = _require_hw(kwargs)
        mask_array = self.to_numpy(h=h, w=w)
        contours, _ = cv2.findContours(mask_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polygons = [Polygon([[x, y] for x, y in contour.reshape(-1, 2)]) for contour in contours]
        return [polygon for polygon in polygons if polygon.encloses_area(**kwargs)]

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
        for polygon in self.to_polygons(h=h, w=w):
            instances.append(polygon.to_yolo(h, w, **kwargs))
        return instances

    def area(self, **kwargs):
        polygons = self.to_polygons(**kwargs)
        area = 0
        for polygon in polygons:
            area += polygon.area(**kwargs)
        return area

    def to_box(self, **kwargs):
        """The single box enclosing the whole mask, disconnected regions included.

        This is what an axis-aligned consumer of one instance wants: a COCO annotation holds its regions
        together under one bbox. Formats that cannot express a disconnected instance want `to_boxes`.
        """
        h, w = _require_hw(kwargs)
        mask_array = self.to_numpy(h=h, w=w)
        boxes = masks_to_boxes(torch.from_numpy(mask_array).unsqueeze(0))
        if boxes is None or boxes.numel() == 0:
            raise ValueError("No boxes found in the mask.")
        return Box(
            x_min=boxes[:, 0].min().item(),
            y_min=boxes[:, 1].min().item(),
            x_max=boxes[:, 2].max().item(),
            y_max=boxes[:, 3].max().item(),
            angle=0,
        )

    def to_boxes(self, **kwargs) -> List[Box]:
        """One box per connected region, for formats where an instance is a single connected shape."""
        return [polygon.to_box(**kwargs) for polygon in self.to_polygons(**kwargs)]


@dataclass
class Label(Base):
    id: str
    color: Optional[str] = None
    annotation_type: AnnotationType = None
    keypoints: Optional[List[str]] = None

    @classmethod
    def from_dict(cls, data: dict) -> "Label":
        return cls(
            id=data["id"],
            color=data.get("color"),
            annotation_type=data.get("annotation_type"),
            keypoints=data.get("keypoints"),
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


def _annotation_encloses_area(annotation: Annotation, h: int, w: int) -> bool:
    """Whether an annotation covers any pixels; geometry that covers none cannot be trained on."""
    if annotation.type == AnnotationType.MASK:
        return bool(annotation.value.to_polygons(h=h, w=w))
    if annotation.type == AnnotationType.POLYGON:
        return annotation.value.encloses_area(h=h, w=w)
    if annotation.type == AnnotationType.BOX:
        return annotation.value.to_polygon().encloses_area(h=h, w=w)
    return True


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
        boxes_by_id = {annotation.id: annotation for annotation in self.annotations if annotation.type == AnnotationType.BOX}
        for annotation in self.annotations:
            if annotation.type != AnnotationType.KEYPOINT:
                continue
            if annotation.bounding_box_id is not None:
                # Exporters provide authoritative links; containment is only a fallback for standalone conversions.
                if annotation.bounding_box_id not in boxes_by_id:
                    raise ValueError(f"Bounding box {annotation.bounding_box_id} not found for keypoint {annotation.id}")
                continue

            candidates = [
                box
                for box in boxes_by_id.values()
                if (not target_ids or box.label_id in target_ids) and box.value.point_in_box(annotation.value.x, annotation.value.y)
            ]
            if len(candidates) == 1:
                annotation.bounding_box_id = candidates[0].id
            elif not candidates:
                raise ValueError(f"Keypoint {annotation.id} not assigned to any box")
            else:
                raise ValueError(f"Keypoint {annotation.id} is contained by multiple boxes")

        return self

    def to_yolo(
        self,
        label_id_idx: dict,
        to_segmentation=False,
        to_object_detection=False,
        merge_boxes=False,
        target_classes=None,
        use_obb=False,
        keypoint_layouts: Optional[Dict[str, List[str]]] = None,
        n_kpts: int = 0,
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

            if not _annotation_encloses_area(annotation, h, w):
                logger.warning(f"Skipping {annotation.type.value} annotation {annotation.id}, which encloses no area")
                continue

            # Conversion steps:
            if annotation.type == AnnotationType.BOX and to_segmentation:
                logger.debug(f"Converting box {annotation.id} to YOLO format with mask_type=AnnotationType.MASK")
                updated_annotations.append(annotation.value.to_mask(h=h, w=w))
            elif annotation.type == AnnotationType.MASK and to_object_detection:
                logger.debug(f"Converting mask {annotation.id} to YOLO format with mask_type=AnnotationType.MASK")
                # a YOLO instance is one connected shape, so each region becomes its own box
                updated_annotations.extend([annotation.value.to_box(h=h, w=w)] if merge_boxes else annotation.value.to_boxes(h=h, w=w))
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

        if n_kpts:
            self.assign_keypoints(target_ids=target_classes)
            keypoint_layouts = keypoint_layouts or {}
            keypoints_by_box = {}
            for annotation in self.annotations:
                if annotation.type != AnnotationType.KEYPOINT or annotation.bounding_box_id not in yolo_annotations_map:
                    continue
                points_by_label = keypoints_by_box.setdefault(annotation.bounding_box_id, {})
                if annotation.label_id in points_by_label:
                    raise ValueError(f"Duplicate keypoint label {annotation.label_id} for bounding box {annotation.bounding_box_id}")
                points_by_label[annotation.label_id] = annotation

            boxes_by_id = {annotation.id: annotation for annotation in self.annotations if annotation.type == AnnotationType.BOX}
            for box_id, row in yolo_annotations_map.items():
                box_annotation = boxes_by_id.get(box_id)
                if box_annotation is None:
                    continue
                layout = keypoint_layouts.get(box_annotation.label_id, [])
                points_by_label = keypoints_by_box.get(box_id, {})
                unknown_labels = set(points_by_label) - set(layout)
                if unknown_labels:
                    labels = ", ".join(sorted(unknown_labels))
                    raise ValueError(f"Keypoint labels {labels} are not in the layout for bounding box {box_id}")
                # Layout position is the YOLO slot index; annotation order must not affect the row.
                for label_id in layout:
                    point = points_by_label.get(label_id)
                    if point is None:
                        row.extend([0, 0, 0])
                    else:
                        visibility = point.value.visibility if point.value.visibility is not None else 2
                        row.extend([point.value.x / w, point.value.y / h, visibility])
                # Every class uses the dataset-wide maximum so all pose rows have the same width.
                for _ in range(n_kpts - len(layout)):
                    row.extend([0, 0, 0])

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
            delete_ids = [label.id for label in self.labels if label.id not in target_classes]
            self.labels = [label for label in self.labels if label.id in target_classes]
            for file_ann in self.files:
                for list_type in ("annotations", "predictions"):
                    annotations = file_ann._get_target_list(list_type)
                    kept_box_ids = {ann.id for ann in annotations if ann.type == AnnotationType.BOX and ann.label_id in target_classes}
                    # Keypoint labels differ from box classes, so retain them according to their owning box.
                    filtered = [
                        ann
                        for ann in annotations
                        if ann.label_id in target_classes
                        or (
                            ann.type == AnnotationType.KEYPOINT
                            and (getattr(ann, "bounding_box_id", None) is None or ann.bounding_box_id in kept_box_ids)
                        )
                    ]
                    file_ann.update_annotations(filtered, list_type=list_type)
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
                    if (
                        annotation.type != AnnotationType.KEYPOINT
                        and annotation.label_id in target_label_ids
                        and annotation.label_id not in label_id_index
                    ):
                        label_id_index[annotation.label_id] = label_idx
                        label_idx += 1

        # sort the label_id_index by label id
        label_id_index = dict(sorted(label_id_index.items(), key=lambda item: item[1]))

        keypoint_layouts = {label.id: label.keypoints for label in self.labels if label.keypoints is not None}
        # Layout length is per instance; counting annotations would vary with images and object counts.
        n_kpts = max((len(layout) for layout in keypoint_layouts.values()), default=0)
        image_to_labels = {}
        label_ids = []
        for file_ann in self.files:
            file_path = file_ann.path

            logger.debug(f"Processing file {file_path}")
            if file_path not in image_to_labels:
                image_to_labels[file_path] = []

            # Call the file-level to_yolo method:
            file_yolo, file_label_ids = file_ann.to_yolo(
                label_id_idx=label_id_index,
                to_segmentation=to_segmentation,
                to_object_detection=to_object_detection,
                merge_boxes=merge_boxes,
                target_classes=target_label_ids,
                use_obb=kwargs.get("use_obb", False),
                keypoint_layouts=keypoint_layouts,
                n_kpts=n_kpts,
            )
            label_ids.extend(file_label_ids)
            image_to_labels[file_path].extend(file_yolo)

        # generate the class map
        return dict(
            image_labels=image_to_labels,
            class_map=label_id_index,
            n_kpts=n_kpts,
        )
