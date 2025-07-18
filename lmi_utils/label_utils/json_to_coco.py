from dataset_utils.representations import Dataset, PolygonAnnotation, MaskAnnotation, BoxAnnotation
from dataset_utils.coco_dataset import CocoDataset, CocoImage, CocoAnnotation
from dataset_utils.file_utils import load_and_update
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

def create_coco_dataset(dataset: Dataset, is_crowd:bool = False, target_classes:list = []) -> CocoDataset:
    """
    Create a COCO dataset from a given dataset and image path.

    Args:
        dataset (Dataset): The dataset to convert.
        path_imgs (str): The path to the images directory.

    Returns:
        CocoDataset: A COCO dataset with updated file dimensions.
    """
    coco_dataset = CocoDataset()
    annotation_id = 0
    # add categories
    labels = dataset.labels
    for label_id, label in enumerate(labels):
        coco_dataset.get_category_by_name(id=label_id+1, name=label.name, supercategory="")

    
    
    # add images and annotations
    for file_id, file in enumerate(dataset.files):
        filtered_annotations = [ann for ann in file.annotations if ann.label.name in target_classes]
        if len(filtered_annotations) == 0:
            logger.warning(f'Skipping file {file.path} as it has no annotations for target classes: {target_classes}')
            continue
        logger.info(f'Processing file {file_id+1}/{len(dataset.files)}: {file.path}')
        image_id = file_id + 1
        annotation_id += 1
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
    return coco_dataset

def main(args):
    path_train_imgs = args['path_train_imgs']
    path_val_imgs = args['path_val_imgs'] if args.get('path_val_imgs') else path_train_imgs
    path_train_json = args['path_train_json'] if args['path_train_json']!='labels.json' else os.path.join(path_train_imgs, args['path_train_json'])
    path_val_json = args['path_val_json'] if args['path_val_json']!='labels.json' else os.path.join(path_val_imgs, args['path_val_json'])
    path_out = args['path_out']
    
    if not os.path.exists(path_out):
        os.makedirs(path_out)
    
    use_train_for_val = path_train_imgs == path_val_imgs and path_train_json == path_val_json
    # load datasets
    train_dataset = load_and_update(annotations_path=path_train_json, path_imgs=path_train_imgs)
    if use_train_for_val:
        logger.info('Using train dataset for validation')
        val_dataset = train_dataset
    else:
        val_dataset = load_and_update(annotations_path=path_val_json, path_imgs=path_val_imgs)

    if not use_train_for_val:
        val_dataset = load_and_update(val_dataset, path_val_imgs)
    # filter target classes
    target_classes = args['target_classes']
    if target_classes != 'all':
        target_classes = [c.strip() for c in target_classes.split(',')]
    else:
        target_classes = [c.name for c in train_dataset.labels]
    logger.info(f'Target classes: {target_classes}')

        
    # create coco datasets
    coco_train_dataset = create_coco_dataset(train_dataset, is_crowd=False, target_classes=target_classes)
    if use_train_for_val:
        logger.info('Creating validation dataset from train dataset')
        coco_val_dataset = coco_train_dataset
    else:
        logger.info('Creating validation dataset from val dataset')
        coco_val_dataset = create_coco_dataset(val_dataset, is_crowd=False, target_classes=target_classes)

    # save coco datasets
    coco_train_dataset.save_to_json(file_path=os.path.join(path_out, 'train.annotations.json'))
    if not use_train_for_val:
        coco_val_dataset.save_to_json(file_path=os.path.join(path_out, 'validation.annotations.json'))
    # TODO: save images

if __name__ == "__main__":
    args = get_args()
    main(args)


            
        
    


