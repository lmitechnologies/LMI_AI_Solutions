from dataset_utils.representations import Dataset, PolygonAnnotation, MaskAnnotation, BoxAnnotation
from dataset_utils.coco_dataset import CocoDataset, CocoImage, CocoAnnotation, CocoCategory
from dataset_utils.file_utils import load_and_update, copy_images_in_folder
import argparse
import os
import logging


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--path_train_json', default='labels.json', help='[optional] the path of a json file for train' )
    ap.add_argument('--path_val_json', default='labels.json', help='[optional] the path of a json file for val')
    ap.add_argument('--path_out', '-o', required=True, help='the output path for dataset')
    ap.add_argument('--path_train_imgs', '-ti', required=True, help='the output path for train images')
    ap.add_argument('--path_val_imgs', '-vi', required=False, help='the output path for val images')
    ap.add_argument('--target_classes',default='all', help='[optional] the comma separated target classes, default=all')
    ap.add_argument('--bg', action='store_true', help='save images with no labels, where yolo models treat them as background')
    ap.add_argument('--merge_box', action='store_true', help='merge multiple instances of same class boxes into one. Brush labels only!')
    args = vars(ap.parse_args())
    return args

        
def get_coco_annotation(annotation, **kwargs):
    """Convert a dataset annotation to COCO format."""
    if isinstance(annotation, MaskAnnotation):
        bbox = annotation.value.to_box(**kwargs)
        return annotation.value.to_coco(**kwargs), bbox.to_coco()
    elif isinstance(annotation, PolygonAnnotation):
        bbox = annotation.value.to_box()
        return annotation.value.to_coco(), bbox.to_coco()
    elif isinstance(annotation, BoxAnnotation):
        poly = annotation.value.to_polygon()
        return poly.to_coco(), annotation.value.to_coco()
    else:
        raise ValueError(f"Unsupported annotation type: {type(annotation)}")

def create_coco_dataset(dataset: Dataset, is_crowd:bool = False, target_classes:list = [], **kwargs) -> CocoDataset:
    """
    Create a COCO dataset from a given dataset and image path.

    Args:
        dataset (Dataset): The dataset to convert.
        path_imgs (str): The path to the images directory.

    Returns:
        CocoDataset: A COCO dataset with updated file dimensions.
    """
    coco_dataset = CocoDataset()
    file_id_map = {}
    annotation_id = 0
    # add categories
    labels = dataset.labels
    for label_id, label in enumerate(labels):
        coco_dataset.add_category(CocoCategory(
            id=label_id + 1,
            name=label.id,
            supercategory='',
        ))
    
    fnames = set()
    # add images and annotations
    for file_id, file in enumerate(dataset.files):
        filtered_annotations = [ann for ann in file.annotations if ann.label_id in target_classes]
        file_id_map[os.path.basename(file.path)] = file.id
        if len(filtered_annotations) == 0:
            logger.warning(f'Skipping file {file.path} as it has no annotations for target classes')
            continue
        
        logger.info(f'Processing file {file_id+1}/{len(dataset.files)}: {file.path}')
        image_id = file_id + 1
        # update the image name if id is not part of the image
        out_name = os.path.basename(file.path)
        if f'id{file.id}_' not in out_name:
            out_name = f'id{file.id}_{out_name}'
        coco_dataset.add_image(CocoImage(
            id=image_id,
            file_name=out_name,
            height=file.height,
            width=file.width,
        ))
        added_annotations = 0
        for annotation in file.annotations:
            try:
                annotation_id += 1
                # both segmentation and bbox are required for COCO format
                segmentation, bbox = get_coco_annotation(annotation, h=file.height, w=file.width, **kwargs)
                if bbox[2] <= 0 or bbox[3] <= 0:
                    logger.warning(f'Skipping annotation {annotation.id} for file {file.path} as bbox is invalid: {bbox}')
                    continue
                coco_dataset.add_annotation(CocoAnnotation(
                    id=annotation_id,
                    image_id=image_id,
                    category_id=coco_dataset.get_category_by_name(annotation.label_id).id,
                    segmentation=segmentation,
                    bbox=bbox,
                    area=0,
                    iscrowd=is_crowd,
                ))
                added_annotations += 1
            except Exception as e:
                logger.error(f'Error processing  (annotation could be invalid) {annotation.id} for file {file.path}: {e}')
                continue
        if added_annotations > 0:
            fnames.add(os.path.basename(file.path))
        else:
            # remove the image if no annotations were added
            logger.warning(f'No valid annotations found for file {file.path}, removing image from dataset')
            for coco_image in coco_dataset.images:
                if coco_image.file_name == out_name:
                    coco_dataset.images.remove(coco_image)
                    # removing annotations for this image
                    break
    return coco_dataset, fnames,file_id_map


def convert_to_json(args):
    path_train_imgs = args['path_train_imgs']
    path_val_imgs = args['path_val_imgs'] if args.get('path_val_imgs') else path_train_imgs
    path_train_json = args['path_train_json'] if args['path_train_json'] !='labels.json' else os.path.join(path_train_imgs, args['path_train_json'])
    path_val_json = args['path_val_json'] if args['path_val_json']!='labels.json' else os.path.join(path_val_imgs, args['path_val_json'])
    path_out = args['path_out']
    background = args.get('bg', False)
    if background:
        logger.warning(f'Background is not supported for COCO format at the moment')
    merge_box = args.get('merge_box', False)
    

    if not os.path.exists(path_train_json):
        raise FileNotFoundError(f'Train annotations file {path_train_json} does not exist')
    
    if not os.path.exists(path_val_json):
        logger.warning(f'Validation annotations file {path_val_json} does not exist, using train annotations instead')
        path_val_json = path_train_json
        path_val_imgs = path_train_imgs
    
    if not os.path.exists(path_out):
        os.makedirs(path_out)
        
    logger.info(f'Train images path: {path_train_imgs}')
    logger.info(f'Validation images path: {path_val_imgs}')
    logger.info(f'Train annotations path: {path_train_json}')
    logger.info(f'Validation annotations path: {path_val_json}')
    use_train_for_val = (path_train_imgs == path_val_imgs and path_train_json == path_val_json)
    logger.info(f'Using train dataset for validation: {use_train_for_val}')
    # load datasets
    train_dataset = load_and_update(annotations_path=path_train_json, path_imgs=path_train_imgs)
    if use_train_for_val:
        logger.info('Using train dataset for validation')
        val_dataset = train_dataset
    else:
        val_dataset = load_and_update(annotations_path=path_val_json, path_imgs=path_val_imgs)

    # filter target classes
    target_classes = args['target_classes']
    if target_classes != 'all':
        target_classes = [c.strip() for c in target_classes.split(',')]
    else:
        target_classes = [c.id for c in train_dataset.labels]
    logger.info(f'Target classes: {target_classes}')

        
    # create coco datasets
    coco_train_dataset, train_files,train_file_id_map = create_coco_dataset(train_dataset, is_crowd=False, target_classes=target_classes, merge_boxes=merge_box)
    if use_train_for_val:
        logger.info('Creating validation dataset from train dataset')
        val_file_id_map = train_file_id_map
        coco_val_dataset = coco_train_dataset
    else:
        logger.info('Creating validation dataset from val dataset')
        coco_val_dataset,val_files,val_file_id_map = create_coco_dataset(val_dataset, is_crowd=False, target_classes=target_classes, merge_boxes=merge_box)

    # save coco datasets
    coco_train_dataset.save_to_json(file_path=os.path.join(path_out, 'train', 'annotations.json'))
    if not use_train_for_val:
        coco_val_dataset.save_to_json(file_path=os.path.join(path_out, 'val','annotations.json'))
    
    # save the train dataset
    copy_images_in_folder(path_train_imgs, os.path.join(path_out, 'train', 'images'), train_files,train_file_id_map)
    if not use_train_for_val:
        # save the val dataset
        copy_images_in_folder(path_val_imgs, os.path.join(path_out, 'val', 'images'),  val_files, val_file_id_map)

def main():
    args = get_args()
    convert_to_json(args)


if __name__ == "__main__":
    main()


            
        
    


