from dataset_utils.representations import Dataset, PolygonAnnotation, MaskAnnotation, BoxAnnotation
from dataset_utils.coco_dataset import CocoDataset, CocoImage, CocoAnnotation
from dataset_utils.file_utils import update_file_dimensions
import os

        
def get_coco_annotation(annotation, **kwargs):
    """Convert a dataset annotation to COCO format."""
    if isinstance(annotation, MaskAnnotation):
        bbox = annotation.to_box(h=kwargs.get("h"), w=kwargs.get("w"), merge_boxes=True)
        return annotation.to_coco(h=kwargs.get("h"), w=kwargs.get("w")), bbox.to_coco()
    elif isinstance(annotation, PolygonAnnotation):
        box = annotation.to_box(h=kwargs.get("h"), w=kwargs.get("w"), merge_boxes=True)
        return annotation.to_coco(), box.to_coco()
    elif isinstance(annotation, BoxAnnotation):
        poly = annotation.to_polygon()
        return poly.to_coco(), annotation.to_coco()
    else:
        raise ValueError(f"Unsupported annotation type: {type(annotation)}")

def create_coco_dataset(dataset: Dataset, path_imgs: str) -> CocoDataset:
    """
    Create a COCO dataset from a given dataset and image path.

    Args:
        dataset (Dataset): The dataset to convert.
        path_imgs (str): The path to the images directory.

    Returns:
        CocoDataset: A COCO dataset with updated file dimensions.
    """
    coco_dataset = CocoDataset()
    labels = dataset.labels
    for label_id, label in enumerate(labels):
        coco_dataset.get_category_by_name(id=label_id+1, name=label.name, supercategory="")
    for file_id, file in enumerate(dataset.files):
        coco_dataset.add_image(CocoImage(
            id=file_id,
            file_name=os.path.basename(file.path),
            height=file.height,
            width=file.width,
        ))
        for annotation in file.annotations:
            segmentation, bbox = get_coco_annotation(annotation, h=file.height, w=file.width)
            coco_dataset.add_annotation(CocoAnnotation(
                id=len(coco_dataset.annotations) + 1,
                image_id=file_id,
                category_id=annotation.label.id + 1,
                segmentation=segmentation,
                bbox=bbox,
                area=annotation.area,
                iscrowd=0 if isinstance(annotation, (PolygonAnnotation, BoxAnnotation)) else 1,
            ))
    return coco_dataset



            
        
    


