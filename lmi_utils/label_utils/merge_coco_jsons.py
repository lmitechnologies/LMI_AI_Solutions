from pycocotools.coco import COCO
import json
from pathlib import Path


def merge_coco_datasets(datasets:list[COCO]) -> COCO:
    merged = COCO()
    merged.dataset['images'] = []
    merged.dataset['annotations'] = []
    merged.dataset['categories'] = []
    
    merged_category = {}
    category_id = 1
    image_id = 1
    annotation_id = 1
    for dataset in datasets:
        # update category ids
        for category in dataset.dataset['categories']:
            name, supercategory = category['name'], category['supercategory']
            if name not in merged_category:
                merged_category[name] = {'id': category_id, 'name': name, 'supercategory': supercategory}
                category_id += 1
                
        # update images
        old_to_new_img_id = {}
        for image in dataset.dataset['images']:
            old_to_new_img_id[image['id']] = image_id
            image['id'] = image_id
            merged.dataset['images'].append(image)
            image_id += 1
        
        # update annotations
        id_to_name = {category['id']: category['name'] for category in dataset.dataset['categories']}
        for annotation in dataset.dataset['annotations']:
            name = id_to_name[annotation['category_id']]
            annotation['id'] = annotation_id
            annotation['image_id'] = old_to_new_img_id[annotation['image_id']]
            # assign to new category id
            annotation['category_id'] = merged_category[name]['id']
            merged.dataset['annotations'].append(annotation)
            annotation_id += 1
            
    # save categories
    for dt in merged_category.values():
        merged.dataset['categories'].append(dt)
    return merged



if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description='Merge multiple COCO datasets into one.')
    ap.add_argument('--datasets', '-d', nargs='+', required=True, help='paths to the COCO dataset json to merge')
    ap.add_argument('--output', '-o', required=True, type=Path, help='path to save the merged COCO dataset json')
    args = ap.parse_args()

    # load COCO annotations
    coco_list = []
    for dataset_path in args.datasets:
        if not Path(dataset_path).exists():
            raise FileNotFoundError(f"Dataset file {dataset_path} does not exist.")
        coco = COCO(dataset_path)
        coco_list.append(coco)

    # merge COCO datasets
    merged = merge_coco_datasets(coco_list)

    # create output directory if it doesn't exist
    args.output.mkdir(parents=True, exist_ok=True)

    # write merged dataset to file
    with open(args.output / 'annotations.json', 'w') as f:
        json.dump(merged.dataset, f, indent=4)
        