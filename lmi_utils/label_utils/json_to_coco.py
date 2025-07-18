from dataset_utils.representations import Dataset, PolygonAnnotation, MaskAnnotation, BoxAnnotation
from dataset_utils.coco_dataset import CocoDataset, CocoImage, CocoAnnotation
from dataset_utils.file_utils import update_file_dimensions
import argparse
import os
import logging

logger = logging.getLogger(__name__)

def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--path_train_json', default='labels.json', help='[optional] the path of a json file for train')
    ap.add_argument('--path_val_json', default='labels.json', help='[optional] the path of a json file for val')
    ap.add_argument('--path_out', '-o', required=True, help='the output path for dataset')
    ap.add_argument('--path_train_imgs', '-ti', required=True, help='the output path for train images')
    ap.add_argument('--path_val_imgs', '-vi', required=False, help='the output path for val images')
    ap.add_argument('--target_classes',default='all', help='[optional] the comma separated target classes, default=all')  
    args = vars(ap.parse_args())
    return args

        
def get_coco_annotation(annotation, **kwargs):
    """Convert a dataset annotation to COCO format."""
    if isinstance(annotation, MaskAnnotation):
        bbox = annotation.to_box(**kwargs)
        return annotation.to_coco(**kwargs), bbox.to_coco()
    elif isinstance(annotation, PolygonAnnotation):
        bbox = annotation.to_box()
        return annotation.to_coco(), bbox.to_coco()
    elif isinstance(annotation, BoxAnnotation):
        poly = annotation.to_polygon()
        return poly.to_coco(), annotation.to_coco()
    else:
        raise ValueError(f"Unsupported annotation type: {type(annotation)}")

def create_coco_dataset(dataset: Dataset, is_crowd:bool = False) -> CocoDataset:
    """
    Create a COCO dataset from a given dataset and image path.

    Args:
        dataset (Dataset): The dataset to convert.
        path_imgs (str): The path to the images directory.

    Returns:
        CocoDataset: A COCO dataset with updated file dimensions.
    """
    coco_dataset = CocoDataset()
    annotation_id = 1
    # add categories
    labels = dataset.labels
    for label_id, label in enumerate(labels):
        coco_dataset.get_category_by_name(id=label_id+1, name=label.name, supercategory="")
    
    # add images and annotations
    for file_id, file in enumerate(dataset.files):
        logger.info(f'Processing file {file_id+1}/{len(dataset.files)}: {file.path}')

        image_id = file_id + 1

        coco_dataset.add_image(CocoImage(
            id=image_id,
            file_name=os.path.basename(file.path),
            height=file.height,
            width=file.width,
        ))

        for annotation in file.annotations:
            # both segmentation and bbox are required for COCO format
            segmentation, bbox = get_coco_annotation(annotation, h=file.height, w=file.width)
            coco_dataset.add_annotation(CocoAnnotation(
                id=annotation_id,
                image_id=image_id,
                category_id=coco_dataset.get_category_by_name(annotation.label.name).id,
                segmentation=segmentation,
                bbox=bbox,
                area=annotation.area(),
                iscrowd=int(is_crowd),
            ))
            annotation_id += 1
    return coco_dataset

def main(args):
    path_train_imgs = args['path_train_imgs']
    path_val_imgs = args['path_val_imgs'] if args.get('path_val_imgs') else path_train_imgs
    path_train_json = args['path_train_json'] if args['path_train_json']!='labels.json' else os.path.join(path_train_imgs, args['path_train_json'])
    path_val_json = args['path_val_json'] if args['path_val_json']!='labels.json' else os.path.join(path_val_imgs, args['path_val_json'])
    path_out = args['path_out']

    if not os.path.exists(path_train_imgs):
        raise Exception('The training image path does not exist')
    if not os.path.exists(path_val_imgs) :
        raise Exception('The validation image path does not exist')
    if not os.path.exists(path_train_json):
        raise Exception('The json file does not exist')
    if not os.path.exists(path_val_json):
        raise Exception('The json file does not exist')
    if not os.path.exists(path_out):
        os.makedirs(path_out)
    
    # load datasets
    train_dataset = Dataset.from_json(path_train_json, path_imgs=path_train_imgs)
    val_dataset = Dataset.from_json(path_val_json, path_imgs=path_val_imgs)
    
    # update file dimensions
    train_dataset = update_file_dimensions(train_dataset, path_train_imgs)
    val_dataset = update_file_dimensions(val_dataset, path_val_imgs)
    # filter target classes
    target_classes = args['target_classes']
    if target_classes != 'all':
        target_classes = [c.strip() for c in target_classes.split(',')]
        train_dataset = train_dataset.filter_by_labels(target_classes)
        val_dataset = val_dataset.filter_by_labels(target_classes)
        
    # create coco datasets
    coco_train_dataset = create_coco_dataset(train_dataset, is_crowd=False)
    coco_val_dataset = create_coco_dataset(val_dataset, is_crowd=False)

    # save coco datasets
    coco_train_dataset.save(os.path.join(path_out, 'train.annotations.json'))
    coco_val_dataset.save(os.path.join(path_out, 'val.annotations.json'))
    # TODO: save images

if __name__ == "__main__":
    args = get_args()
    main(args)


            
        
    


