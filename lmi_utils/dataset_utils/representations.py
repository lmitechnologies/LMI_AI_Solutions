from dataclasses import dataclass, asdict
import enum
import json
from typing import Optional, Union, List
import os
import numpy as np
import logging
import cv2
from dataset_utils.mask_encoder import rle2mask, mask2rle
from image_utils.img_resize import resize
from gadget_utils.pipeline_utils import fit_array_to_size
from label_utils.bbox_utils import rotate

logger = logging.getLogger(__name__)

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

    def __init__(self, x: float, y: float):
        super().__init__()
        self.x = float(x)
        self.y = float(y)

    @classmethod
    def from_dict(cls, data: dict) -> "Point2d":
        return cls(x=data["x"], y=data["y"])

    def resize(self, orig_h: int, orig_w: int, new_h: int, new_w: int):
        rx = new_w / orig_w if orig_w else 1
        ry = new_h / orig_h if orig_h else 1
        self.x *= rx
        self.y *= ry
        return self

    def pad(self, **kwargs):
        pl = kwargs.get("pl", 0)
        pt = kwargs.get("pt", 0)
        self.x += pl
        self.y += pt
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

    def __init__(self, x_min, y_min, x_max, y_max, angle=0):
        super().__init__()
        self.x_min = float(x_min)
        self.y_min = float(y_min)
        self.x_max = float(x_max)
        self.y_max = float(y_max)
        self.angle = float(angle)
        if self.x_min > self.x_max:
            raise ValueError("x_min must be less than x_max")
        if self.y_min > self.y_max:
            raise ValueError("y_min must be less than y_max")

    @classmethod
    def from_dict(cls, data: dict) -> "Box":
        return cls(**data)

    def resize(self, orig_h: int, orig_w: int, new_h: int, new_w: int):
        if orig_w <= 0 or orig_h <= 0:
            raise ValueError("Original dimensions must be positive")
        if new_w <= 0 or new_h <= 0:
            raise ValueError("New dimensions must be positive")
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

    def to_numpy(self):
        return np.array([self.x_min, self.y_min, self.x_max, self.y_max, self.angle])

    def coords(self, **kwargs):
        return self.x_min, self.y_min, self.x_max, self.y_max, self.angle

    def to_yolo(self, h, w, **kwargs):
        use_obb = kwargs.get("use_obb", False)
        
        width = self.x_max - self.x_min
        height = self.y_max - self.y_min
        cx = (self.x_min + self.x_max) / 2
        cy = (self.y_min + self.y_max) / 2
        if self.angle > 0 and use_obb:
            
            rotated_coords = rotate(
                self.x_min,
                self.y_min,
                width,
                height,
                self.angle,
                rot_center="up_left",
                unit="degree",
            )
            for p in rotated_coords:
                if p[0] > w:
                    raise ValueError(f"Rotated point x value {p[0]} is greater than image width {w}")
                if p[1] > h:
                    raise ValueError(f"Rotated point y value {p[1]} is greater than image height {h}")
            return [[pt[0] / w, pt[1] / h] for pt in rotated_coords]
        else:
            if use_obb:
                logger.debug(f"Use_obb is True but angle is {self.angle}; returning obb formatted bounding box.")
                corners = np.array([[self.x_min, self.y_min], [self.x_max, self.y_min], [self.x_max, self.y_max], [self.x_min, self.y_max]])
                return [[pt[0] / w, pt[1] / h] for pt in corners]
            else:
                # convert to center_x, center_y, width, height
                return [
                    [cx/w, cy/h, (self.x_max - self.x_min)/w, (self.y_max - self.y_min)/h]
                ]

    def to_mask(self, **kwargs):
        img_h = kwargs.get("h")
        img_w = kwargs.get("w")
        mask_type = kwargs.get("mask_type", AnnotationType.MASK)
        if mask_type == AnnotationType.MASK:
            mask = np.zeros((img_h, img_w), dtype=np.uint8)
            mask[int(self.y_min): int(self.y_max), int(self.x_min): int(self.x_max)] = 1
            return Mask(mask=mask)
        elif mask_type == AnnotationType.POLYGON:
            return Polygon(
                points=[
                    [self.x_min, self.y_min],
                    [self.x_max, self.y_min],
                    [self.x_max, self.y_max],
                    [self.x_min, self.y_max],
                ]
            )
        else:
            raise ValueError("Unsupported mask_type in Box.to_mask")
    
    def point_in_box(self, x: int, y: int):
        return self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max


@dataclass
class Polygon(Base):
    points: Union[List[List[int]], List[List[float]], np.ndarray]
    
    def __init__(self, points: Union[List[List[int]], List[List[float]], np.ndarray] = []):
        super().__init__()
        if isinstance(points, np.ndarray):
            points = points.astype(float).tolist()
        self.points = points

    @classmethod
    def from_dict(cls, data: dict) -> "Polygon":
        return cls(**data)

    def resize(self, orig_h: int, orig_w: int, new_h: int, new_w: int):
        for point in self.points:
            p = Point2d(x=point[0], y=point[1])
            p = p.resize(orig_h, orig_w, new_h, new_w)
            point[0], point[1] = p.x, p.y
        return self

    def pad(self, **kwargs):
        pl = kwargs.get("pl", 0)
        pt = kwargs.get("pt", 0)
        for point in self.points:
            p = Point2d(x=point[0], y=point[1])
            p = p.pad(pl=pl, pt=pt)
            point[0], point[1] = p.x, p.y
        return self

    def to_numpy(self):
        return np.array(self.points)

    def coords(self, **kwargs):
        points = np.array(self.points)
        return points[:, 0].tolist(), points[:, 1].tolist()

    def to_yolo(self, h, w, **kwargs):
        return [[point[0] / w, point[1] / h] for point in self.points]

    def to_mask(self, **kwargs):
        img_h = kwargs.get("h")
        img_w = kwargs.get("w")
        mask = np.zeros((img_h, img_w), dtype=np.uint8)
        pts = self.to_numpy().astype(np.int32)
        cv2.fillPoly(mask, [pts], 1)
        return Mask(mask=mask2rle(mask))


@dataclass
class Mask(Base):
    mask: str

    def __init__(
        self,
        mask: Union[str, np.ndarray],
    ):
        super().__init__()
        if not isinstance(mask, (str, np.ndarray)):
            raise ValueError("Mask must be a string or numpy array")
        if isinstance(mask, np.ndarray):
            self.mask = mask2rle(mask)
        else:
            self.mask = mask
    @classmethod
    def from_dict(cls, data: dict) -> "Mask":
        instance = cls(
            **data
        )
        return instance

    def resize(self, orig_h: int, orig_w: int, new_h: int, new_w: int):
        assert orig_h > 0 and orig_w > 0, "Original height and width must be positive"
        mask_array = rle2mask(self.mask, h=orig_h, w=orig_w)
        resized_mask = resize(mask_array, width=new_w, height=new_h)
        self.mask = mask2rle(resized_mask)
        return self

    def pad(self, **kwargs):
        h = kwargs.get("h", None)
        w = kwargs.get("w", None)
        if h is None or w is None:
            raise ValueError("Height and width cannot be None")
        pad_h = kwargs.get("pad_h", 0)
        pad_w = kwargs.get("pad_w", 0)
        mask_array = rle2mask(self.mask, h=kwargs.get("h"), w=kwargs.get("w"))
        mask_array, _, _, _, _ = fit_array_to_size(mask_array, pad_w, pad_h)
        self.mask = mask2rle(mask_array)
        return self

    def to_numpy(self, **kwargs):
        h = kwargs.get("h", None)
        w = kwargs.get("w", None)
        if h is None or w is None:
            raise ValueError("Height and width cannot be None")
        return rle2mask(self.mask, h, w)

    def coords(self, **kwargs):
        h = kwargs.get("h", None)
        w = kwargs.get("w", None)
        if h is None or w is None:
            raise ValueError("Height and width cannot be None")
        mask = self.to_numpy(h=h, w=w)
        ys, xs = np.nonzero(mask==1)
        return xs.tolist(), ys.tolist()

    def to_polygon(self, **kwargs) -> List[Polygon]:
        h = kwargs.get("h", None)
        w = kwargs.get("w", None)
        if h is None or w is None:
            raise ValueError("Height and width cannot be None")
        mask_array = self.to_numpy(h=h, w=w)
        contours, _ = cv2.findContours(mask_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polygons = [contour.reshape(-1, 2) for contour in contours]
        return [Polygon([[x, y] for x, y in polygon]) for polygon in polygons]

    def to_yolo(self, h, w, **kwargs):
        # Delegate conversion to polygons.
        instances = []
        for polygon in self.to_polygon(h=h, w=w):
            instances.append(polygon.to_yolo(h, w, **kwargs))
        return instances

    def to_box(self, **kwargs):
        h = kwargs.get("h", None)
        w = kwargs.get("w", None)
        if h is None or w is None:
            raise ValueError("Height and width cannot be None")
        merge_boxes = kwargs.get("merge_boxes", False)
        mask_array = self.to_numpy(h=kwargs.get("h"), w=kwargs.get("w"))
        if merge_boxes:
            pts = np.column_stack(np.where(mask_array > 0))
            if pts.size == 0:
                raise ValueError("Mask is empty; cannot compute bounding box.")
            x, y, w_box, h_box = cv2.boundingRect(pts)
            return Box(x_min=x, y_min=y, x_max=x + w_box, y_max=y + h_box, angle=0)
        else:
            boxes = []
            for poly in self.to_polygon(h=kwargs.get("h"), w=kwargs.get("w")):
                xs, ys = poly.coords()
                pts = np.array(list(zip(xs, ys)), dtype=np.int32)
                if pts.size == 0:
                    continue
                x, y, w_box, h_box = cv2.boundingRect(pts)
                boxes.append(Box(x_min=x, y_min=y, x_max=x + w_box, y_max=y + h_box, angle=0))
            return boxes

@dataclass
class Label(Base):
    id: str
    annotation_type: AnnotationType = None
    color: Optional[str] = None

    def __init__(self, id: str, color: Optional[str] = None, annotation_type: AnnotationType = None):
        super().__init__()
        self.id = id
        self.color = color
        self.annotation_type = annotation_type

    @classmethod
    def from_dict(cls, data: dict) -> "Label":
        return cls(id=data["id"], color=data.get("color", None), annotation_type=data.get("annotation_type", None))

@dataclass
class Annotation(Base):
    id: str
    label_id: str
    type: AnnotationType
    value: Union[Box, Mask, Point2d, Polygon] = None
    link: Optional[str] = None,
    confidence: Optional[float] = None,
    iou: Optional[float] = None,

    def __init__(
        self,
        id: str,
        label_id: str,
        type: AnnotationType=None,
        value: Union[Box, Mask, Point2d, Polygon] = None,
        link: Optional[str] = None,
        confidence: Optional[float] = None,
        iou: Optional[float] = None,
    ):
        super().__init__()
        self.id = id
        self.label_id = label_id
        self.type = type
        self.link = link
        self.confidence = confidence
        self.iou = iou
        self.value = value

    @classmethod
    def from_dict(cls, data: dict) -> "Annotation":
        ann_type = data.get("type")
        if ann_type == AnnotationType.BOX.value:
            return BoxAnnotation.from_dict(data)
        elif ann_type == AnnotationType.MASK.value:
            return MaskAnnotation.from_dict(data)
        elif ann_type == AnnotationType.KEYPOINT.value:
            return KeypointAnnotation.from_dict(data)
        elif ann_type == AnnotationType.POLYGON.value:
            return PolygonAnnotation.from_dict(data)
        else:
            raise ValueError(f"Unsupported annotation type: {ann_type}")
    
class BoxAnnotation(Annotation):
    value: Box
    def __init__(
        self,
        id: str,
        label_id: str,
        value: Box,
        link: Optional[str] = None,
        confidence: Optional[float] = None,
        iou: Optional[float] = None,
    ):
        super().__init__(id=id, label_id=label_id, type=AnnotationType.BOX, link=link, confidence=confidence, iou=iou)
        self.value = value

    @classmethod
    def from_dict(cls, data: dict) -> "BoxAnnotation":
        return cls(
            id=data["id"],
            label_id=data["label_id"],
            value=Box.from_dict(data["value"]),
            link=data.get("link"),
            confidence=data.get("confidence", None),
            iou=data.get("iou", None),
        )
        

    def to_yolo(self, h, w, **kwargs):
        return self.value.to_yolo(h, w, **kwargs)

class MaskAnnotation(Annotation):
    value: Mask

    def __init__(
        self,
        id: str,
        label_id: str,
        value: Mask,
        link: Optional[str] = None,
        confidence: Optional[float] = None,
        iou: Optional[float] = None,
    ):
        super().__init__(id=id, label_id=label_id, type=AnnotationType.MASK, link=link, confidence=confidence, iou=iou)
        self.value = value
    
    @classmethod
    def from_dict(cls, data: dict) -> "MaskAnnotation":
        return cls(
            id=data["id"],
            label_id=data["label_id"],
            value=Mask.from_dict(data["value"]),
            link=data.get("link"),
            confidence=data.get("confidence", None),
            iou=data.get("iou", None),
        )

    def to_yolo(self, h, w, **kwargs):
        return self.value.to_yolo(h, w, **kwargs)

class KeypointAnnotation(Annotation):
    value: Point2d

    def __init__(
        self,
        id: str,
        label_id: str,
        value: Point2d,
        link: Optional[str] = None,
        confidence: Optional[float] = None,
        iou: Optional[float] = None,
        bounding_box_id: Optional[str] = None,
    ):
        super().__init__(id=id, label_id=label_id, type=AnnotationType.KEYPOINT, link=link, confidence=confidence, iou=iou)
        self.bounding_box_id = bounding_box_id
        self.value = value

    @classmethod
    def from_dict(cls, data: dict) -> "KeypointAnnotation":
        return cls(
            id=data["id"],
            label_id=data["label_id"],
            value=Point2d.from_dict(data["value"]),
            link=data.get("link"),
            confidence=data.get("confidence", None),
            iou=data.get("iou", None),
            bounding_box_id=data.get("bounding_box_id", None),
        )

    def to_yolo(self, h, w, **kwargs):
        return self.value.to_yolo(h, w, **kwargs)


class PolygonAnnotation(Annotation):
    value: Polygon

    def __init__(
        self,
        id: str,
        label_id: str,
        value: Polygon,
        link: Optional[Union[str, None]] = None,
        confidence: Optional[float] = None,
        iou: Optional[float] = None,
    ):
        super().__init__(id=id, label_id=label_id, type=AnnotationType.POLYGON, link=link, confidence=confidence, iou=iou)
        self.value = value

    @classmethod
    def from_dict(cls, data: dict) -> "PolygonAnnotation":
        return cls(
            id=data["id"],
            label_id=data["label_id"],
            value=Polygon.from_dict(data["value"]),
            link=data.get("link"),
            confidence=data.get("confidence", None),
            iou=data.get("iou", None),
        )

    def to_yolo(self, h, w, **kwargs):
        return self.value.to_yolo(h, w, **kwargs)

@dataclass
class FileAnnotations(Base):
    id: str     # File ID
    path: str   # File path
    height: int # File height
    width: int  # File width
    annotations: List[Annotation]
    predictions: List[Annotation]

    def __init__(
        self,
        id: str,
        path:str,
        height:int,
        width: int,
        annotations: List[Annotation] = [],
        predictions: List[Annotation] = [],
    ):
        super().__init__()
        self.id = id
        self.path = path
        self.height = height
        self.width = width
        self.annotations = annotations
        self.predictions = predictions

    @classmethod
    def from_dict(cls, data: dict) -> "FileAnnotations":
        annotations = [Annotation.from_dict(a) for a in data.get("annotations", [])]
        predictions = [Annotation.from_dict(a) for a in data.get("predictions", [])]
        return cls(id=data['id'],path=data['path'], height=data.get('height', None), width=data.get('width', None), annotations=annotations, predictions=predictions)

    @property
    def has_annotations(self) -> bool:
        return len(self.annotations) > 0

    def relative_path(self, base_path: str) -> str:
        return os.path.relpath(self.path, base_path)

    def update_file(self, id, path,height,width):
        self.id = id
        self.path = path
        self.height = height
        self.width = width
        return self

    def delete_annotation(
        self, annotation_id: str, list_type: str = "annotations"
    ) -> bool:
        if list_type not in ["annotations", "predictions"]:
            raise ValueError("list_type must be either 'annotations' or 'predictions'")
        target_list = self.annotations if list_type == "annotations" else self.predictions
        for index, ann in enumerate(target_list):
            if ann.id == annotation_id:
                del target_list[index]
                logger.debug(f"Deleted annotation with id '{annotation_id}' from {list_type}.")
                return True
        logger.warning(f"Annotation with id '{annotation_id}' not found in {list_type}.")
        return False

    def get_annotations_by_type(
        self, annotation_type: AnnotationType, list_type: str = "annotations"
    ) -> List[Annotation]:
        if list_type not in ["annotations", "predictions"]:
            raise ValueError("list_type must be either 'annotations' or 'predictions'")
        target_list = self.annotations if list_type == "annotations" else self.predictions
        return [ann for ann in target_list if ann.type == annotation_type]

    def update_annotations(
        self, annotations: List[Annotation], list_type: str = "annotations"
    ):
        if list_type not in ["annotations", "predictions"]:
            raise ValueError("list_type must be either 'annotations' or 'predictions'")
        if list_type == "annotations":
            self.annotations = annotations
        else:
            self.predictions = annotations

    def assign_keypoints(self, target_ids=[]):
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

    def to_yolo(self, label_id_idx: dict,to_segmentation=False, to_object_detection=False, merge_boxes=False, target_classes=[], use_obb=False):
        """Convert this file's annotations to YOLO format.
           `label_to_index` is a function mapping a label id to an integer index.
        """
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
            elif (annotation.type == AnnotationType.MASK and to_object_detection):
                logger.debug(f"Converting mask {annotation.id} to YOLO format with mask_type=AnnotationType.MASK")
                updated_annotations.append(annotation.value.to_box(h=h, w=w, merge_boxes=merge_boxes))
            elif (annotation.type == AnnotationType.POLYGON and to_object_detection):
                logger.debug(f"Converting polygon {annotation.id} to YOLO format with mask_type=AnnotationType.POLYGON")
                updated_annotations.append(annotation.value.to_box(h=h, w=w))

            converted = [
                ann.to_yolo(h, w, use_obb=use_obb) for ann in updated_annotations
            ] if updated_annotations else [annotation.to_yolo(h, w, use_obb=use_obb)]
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
                box = yolo_annotations_map.get(annotation.bounding_box_id, None)
                
                if box is None:
                    raise Exception(f"Bounding box {annotation.bounding_box_id} not found for keypoint {annotation.id}")
                idx = yolo_annotations.index(box)
                yolo_kp = annotation.to_yolo(h, w, use_obb=use_obb)
                # Add the keypoint to the box annotation
                box.extend(np.array(yolo_kp).flatten().tolist())
                # Update the box annotation in the list
                yolo_annotations[idx] = box
                
        if len(yolo_annotations) == 0:
            logger.debug(f"No annotations found for file {self.path}")
        return yolo_annotations, label_ids


@dataclass
class Dataset(Base):
    labels: List[Label]
    files: List[FileAnnotations]

    @classmethod
    def from_dict(cls, data: dict) -> "Dataset":
        labels = [Label.from_dict(l) for l in data.get("labels", [])]
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
            label_id_index = {name:idx for idx,name in class_map.items()}
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
                    raise Exception(
                        f"Inconsistent number of keypoints: expected {n_kpts}, found {len(keypoints)}"
                    )
                    

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