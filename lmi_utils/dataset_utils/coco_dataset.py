import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Union


@dataclass
class CocoInfo:
    """
    Represents the 'info' block of the COCO format.
    Contains high-level information about the dataset.
    """

    year: int
    version: str
    description: str
    contributor: str
    url: str = ""
    date_created: Optional[str] = None

    def __post_init__(self):
        if self.date_created is None:
            self.date_created = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def to_dict(self) -> Dict[str, Any]:
        """Converts the Info object to a dictionary."""
        return asdict(self)


@dataclass
class CocoLicense:
    """
    Represents the 'license' block of the COCO format.
    Contains information about a single image license.
    """

    id: int
    name: str
    url: str = ""

    def __post_init__(self):
        if self.id < 0:
            raise ValueError("License ID must be non-negative")

    def to_dict(self) -> Dict[str, Any]:
        """Converts the License object to a dictionary."""
        return asdict(self)


@dataclass
class CocoImage:
    """
    Represents the 'image' block of the COCO format.
    Contains information about a single image.
    """

    id: int
    width: int
    height: int
    file_name: str
    license: int = 0
    flickr_url: Optional[str] = None
    coco_url: Optional[str] = None
    date_captured: Optional[str] = None

    def __post_init__(self):
        if self.id < 0:
            raise ValueError("Image ID must be non-negative")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Image width and height must be positive")
        if not self.file_name:
            raise ValueError("File name cannot be empty")
        if self.license < 0:
            raise ValueError("License ID must be non-negative")

    def to_dict(self) -> Dict[str, Any]:
        """Converts the Image object to a dictionary."""
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class CocoAnnotation:
    """
    Represents the 'annotation' block of the COCO format.
    Contains information about a single annotation.
    The `segmentation` can be a list of polygons or an RLE object.
    """

    id: int
    image_id: int
    category_id: int
    segmentation: Union[List[List[float]], Dict[str, Any]]
    area: float
    bbox: List[float]
    iscrowd: bool = False

    def __post_init__(self):
        if self.id < 0:
            raise ValueError("Annotation ID must be non-negative")
        if self.image_id < 0:
            raise ValueError("Image ID must be non-negative")
        if self.category_id < 0:
            raise ValueError("Category ID must be non-negative")
        if self.area < 0:
            raise ValueError("Area must be non-negative")
        if self.iscrowd not in [0, 1]:
            raise ValueError("iscrowd must be 0 or 1")

        # Validate bbox format [x, y, width, height]
        if not isinstance(self.bbox, list) or len(self.bbox) != 4:
            raise ValueError("bbox must be a list of 4 numbers [x, y, width, height]")
        if any(not isinstance(x, (int, float)) for x in self.bbox):
            raise ValueError("bbox values must be numbers")
        if self.bbox[2] <= 0 or self.bbox[3] <= 0:
            raise ValueError("bbox width and height must be positive")

    def to_dict(self) -> Dict[str, Any]:
        """Converts the Annotation object to a dictionary."""
        return asdict(self)


@dataclass
class CocoCategory:
    """
    Represents the 'category' block of the COCO format.
    Contains information about a single object category.
    """

    id: int
    name: str
    supercategory: str = ""

    def __post_init__(self):
        if self.id < 0:
            raise ValueError("Category ID must be non-negative")
        if not self.name:
            raise ValueError("Category name cannot be empty")

    def to_dict(self) -> Dict[str, Any]:
        """Converts the Category object to a dictionary."""
        return asdict(self)


@dataclass
class CocoDataset:
    """
    A class to represent and manage a dataset in the COCO JSON format.

    The COCO format is structured with the following main keys:
    - info: Contains high-level information about the dataset.
    - licenses: A list of image licenses.
    - images: A list of all images in the dataset.
    - annotations: A list of all annotations (e.g., bounding boxes, masks).
    - categories: A list of all object categories.
    """

    info: Optional[CocoInfo] = field(
        default_factory=lambda: CocoInfo(
            year=datetime.now().year,
            version="1.0",
            description="COCO dataset",
            contributor="Unknown",
            url="",
            date_created=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )
    )
    licenses: List[CocoLicense] = field(default_factory=list)
    images: List[CocoImage] = field(default_factory=list)
    annotations: List[CocoAnnotation] = field(default_factory=list)
    categories: List[CocoCategory] = field(default_factory=list)

    # Private fields to track IDs (not serialized)
    _license_ids: Set[int] = field(default_factory=set, init=False, repr=False)
    _image_ids: Set[int] = field(default_factory=set, init=False, repr=False)
    _annotation_ids: Set[int] = field(default_factory=set, init=False, repr=False)
    _category_ids: Set[int] = field(default_factory=set, init=False, repr=False)

    def set_info(self, info: CocoInfo):
        """Sets the dataset info."""
        if not isinstance(info, CocoInfo):
            raise TypeError("info must be an instance of CocoInfo")
        self.info = info

    def add_license(self, license_obj: CocoLicense):
        """Adds a license to the dataset."""
        if not isinstance(license_obj, CocoLicense):
            raise TypeError("license_obj must be an instance of CocoLicense")
        if license_obj.id in self._license_ids:
            raise ValueError(f"License with ID {license_obj.id} already exists")

        self.licenses.append(license_obj)
        self._license_ids.add(license_obj.id)

    def add_image(self, image: CocoImage):
        """Adds an image to the dataset."""
        if not isinstance(image, CocoImage):
            raise TypeError("image must be an instance of CocoImage")
        if image.id in self._image_ids:
            raise ValueError(f"Image with ID {image.id} already exists")

        # Check if license exists
        if image.license not in self._license_ids and image.license != 0:
            raise ValueError(f"License with ID {image.license} does not exist")

        self.images.append(image)
        self._image_ids.add(image.id)

    def add_annotation(self, annotation: CocoAnnotation):
        """Adds an annotation to the dataset."""
        if not isinstance(annotation, CocoAnnotation):
            raise TypeError("annotation must be an instance of CocoAnnotation")
        if annotation.id in self._annotation_ids:
            raise ValueError(f"Annotation with ID {annotation.id} already exists")

        # Check if image and category exist
        if annotation.image_id not in self._image_ids:
            raise ValueError(f"Image with ID {annotation.image_id} does not exist")
        if annotation.category_id not in self._category_ids:
            raise ValueError(f"Category with ID {annotation.category_id} does not exist")

        self.annotations.append(annotation)
        self._annotation_ids.add(annotation.id)

    def add_category(self, category: CocoCategory):
        """Adds a category to the dataset."""
        if not isinstance(category, CocoCategory):
            raise TypeError("category must be an instance of CocoCategory")
        if category.id in self._category_ids:
            raise ValueError(f"Category with ID {category.id} already exists")

        self.categories.append(category)
        self._category_ids.add(category.id)

    def get_image_by_id(self, image_id: int) -> Optional[CocoImage]:
        """Returns an image by its ID."""
        for image in self.images:
            if image.id == image_id:
                return image
        return None

    def get_category_by_id(self, category_id: int) -> Optional[CocoCategory]:
        """Returns a category by its ID."""
        for category in self.categories:
            if category.id == category_id:
                return category
        return None

    def get_category_by_name(self, category_name: str) -> Optional[CocoCategory]:
        """Returns a category by its name."""
        for category in self.categories:
            if category.name == category_name:
                return category
        return None

    def get_categories_by_name(self, category_name: str) -> List[CocoCategory]:
        """Returns all categories that match the given name (in case of duplicates)."""
        return [category for category in self.categories if category.name == category_name]

    def get_annotations_by_image_id(self, image_id: int) -> List[CocoAnnotation]:
        """Returns all annotations for a specific image."""
        return [ann for ann in self.annotations if ann.image_id == image_id]

    def get_annotations_by_category_id(self, category_id: int) -> List[CocoAnnotation]:
        """Returns all annotations for a specific category."""
        return [ann for ann in self.annotations if ann.category_id == category_id]

    def get_annotations_by_category_name(self, category_name: str) -> List[CocoAnnotation]:
        """Returns all annotations for a specific category name."""
        category = self.get_category_by_name(category_name)
        if category is None:
            return []
        return self.get_annotations_by_category_id(category.id)

    def validate_dataset(self) -> List[str]:
        """
        Validates the dataset and returns a list of validation errors.
        Returns an empty list if the dataset is valid.
        """
        errors = []

        # Check if info is set
        if self.info is None:
            errors.append("Dataset info is not set")

        # Check if we have at least one category
        if not self.categories:
            errors.append("Dataset must have at least one category")

        # Check for duplicate category names
        category_names = [cat.name for cat in self.categories]
        duplicate_names = set([name for name in category_names if category_names.count(name) > 1])
        if duplicate_names:
            errors.append(f"Warning: Duplicate category names found: {', '.join(duplicate_names)}")

        # Check for orphaned annotations
        for annotation in self.annotations:
            if annotation.image_id not in self._image_ids:
                errors.append(f"Annotation {annotation.id} references non-existent image {annotation.image_id}")
            if annotation.category_id not in self._category_ids:
                errors.append(f"Annotation {annotation.id} references non-existent category {annotation.category_id}")

        # Check for images without annotations (warning, not error)
        images_with_annotations = {ann.image_id for ann in self.annotations}
        for image in self.images:
            if image.id not in images_with_annotations:
                errors.append(f"Warning: Image {image.id} has no annotations")

        return errors

    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the entire CocoDataset object to a dictionary
        adhering to the COCO JSON structure.
        """
        return {
            "info": self.info.to_dict() if self.info else {},
            "licenses": [lic.to_dict() for lic in self.licenses],
            "images": [img.to_dict() for img in self.images],
            "annotations": [ann.to_dict() for ann in self.annotations],
            "categories": [cat.to_dict() for cat in self.categories],
        }

    def save_to_json(self, file_path: str, indent: int = 4, validate: bool = True):
        """
        Saves the dataset to a JSON file.

        Args:
            file_path (str): The path to the output JSON file.
            indent (int): The indentation level for the JSON output.
            validate (bool): Whether to validate the dataset before saving.

        Raises:
            ValueError: If validation fails and validate=True.
            IOError: If the file cannot be written.
        """
        if validate:
            errors = self.validate_dataset()
            if any(not error.startswith("Warning:") for error in errors):
                raise ValueError(f"Dataset validation failed: {'; '.join(errors)}")

        # Ensure the directory exists
        if os.path.dirname(file_path):
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
        try:
            with open(file_path, "w") as f:
                json.dump(self.to_dict(), f, indent=indent)
        except IOError as e:
            raise IOError(f"Could not write to file {file_path}: {e}") from None

    @classmethod
    def load_from_json(cls, file_path: str) -> "CocoDataset":
        """
        Loads a COCO dataset from a JSON file.

        Args:
            file_path (str): Path to the JSON file.

        Returns:
            CocoDataset: The loaded dataset.
        """
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
        except (IOError, json.JSONDecodeError) as e:
            raise ValueError(f"Could not load JSON file {file_path}: {e}") from None

        dataset = cls()

        # Load info
        if "info" in data and data["info"]:
            info_data = data["info"]
            dataset.set_info(
                CocoInfo(
                    year=info_data.get("year", 0),
                    version=info_data.get("version", ""),
                    description=info_data.get("description", ""),
                    contributor=info_data.get("contributor", ""),
                    url=info_data.get("url", ""),
                    date_created=info_data.get("date_created"),
                )
            )

        # Load licenses
        for lic_data in data.get("licenses", []):
            dataset.add_license(CocoLicense(id=lic_data["id"], name=lic_data["name"], url=lic_data.get("url", "")))

        # Load categories
        for cat_data in data.get("categories", []):
            dataset.add_category(CocoCategory(id=cat_data["id"], name=cat_data["name"], supercategory=cat_data.get("supercategory", "")))

        # Load images
        for img_data in data.get("images", []):
            dataset.add_image(
                CocoImage(
                    id=img_data["id"],
                    width=img_data["width"],
                    height=img_data["height"],
                    file_name=img_data["file_name"],
                    license=img_data.get("license", 0),
                    flickr_url=img_data.get("flickr_url"),
                    coco_url=img_data.get("coco_url"),
                    date_captured=img_data.get("date_captured"),
                )
            )

        # Load annotations
        for ann_data in data.get("annotations", []):
            dataset.add_annotation(
                CocoAnnotation(
                    id=ann_data["id"],
                    image_id=ann_data["image_id"],
                    category_id=ann_data["category_id"],
                    segmentation=ann_data["segmentation"],
                    area=ann_data["area"],
                    bbox=ann_data["bbox"],
                    iscrowd=ann_data.get("iscrowd", False),
                )
            )

        return dataset

    def get_statistics(self) -> Dict[str, Any]:
        """Returns basic statistics about the dataset."""
        return {
            "num_images": len(self.images),
            "num_annotations": len(self.annotations),
            "num_categories": len(self.categories),
            "num_licenses": len(self.licenses),
            "annotations_per_image": len(self.annotations) / len(self.images) if self.images else 0,
            "categories": [cat.name for cat in self.categories],
        }

    def __post_init__(self):
        """Initialize ID tracking sets after dataclass initialization."""
        # Rebuild ID tracking sets from existing data
        self._license_ids = {lic.id for lic in self.licenses}
        self._image_ids = {img.id for img in self.images}
        self._annotation_ids = {ann.id for ann in self.annotations}
        self._category_ids = {cat.id for cat in self.categories}
