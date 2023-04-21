import os
import argparse
import logging
import json
from label_utils.COCO_dataset import COCO_Dataset, Annotation, rotate

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def convert_from_ls(result):
    """convert annotations from label studio format to image pixel coordinate system

    Returns:
        tuple: x,y,w,h,angle, where angle is counterclockwise rotation angle in degree
    """
    value = result['value']
    w, h = result['original_width'], result['original_height']
    
    if not all([key in value for key in ['x', 'y', 'width', 'height']]):
        raise Exception('missing "x", "y", "width", or "height" in json file')
    
    # angle in degree
    angle = value['rotation'] if 'rotation' in value else 0
    
    # convert angle from anticlockwise to clockwise
    if angle>180:
        angle -= 360
        
    if angle>45 or angle<-45:
        logging.warning(f'found angle of {angle} out of constrains')
    
    return w * value['x'] / 100.0, h * value['y'] / 100.0, \
           w * value['width'] / 100.0, h * value['height'] / 100.0, \
           angle


def get_annotations_from_json(path_json, path_imgs, plot=False):
    annots = []
    dt_category = {}
    id = 1
    with open(path_json) as f:
        l = json.load(f)
        for dt in l:
            # get rid of first '-'
            f = dt['file_upload'].split('-')[1:]
            fname = '-'.join(f)
            logger.info(fname)
            path_img = os.path.join(path_imgs,fname)
            if not os.path.isfile(path_img):
                raise Exception(f'Cannot find the file: {path_img}')
            
            for annot in dt['annotations']:
                for result in annot['result']:
                    # assume each annotation has only one label
                    label = result['value']['rectanglelabels'][0]
                    if label not in dt_category:
                        dt_category[label] = id
                        id += 1
                    x,y,w,h,angle = convert_from_ls(result)
                    # convert to int
                    x,y,w,h = list(map(int,[x,y,w,h]))
                    pts = rotate(x,y,w,h,angle)
                    annot = Annotation(path_img,label,bbox=[x,y,w,h],rotation=angle,segmentation=pts)
                    annots.append(annot)
    return annots,dt_category


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--path_imgs', required=True)
    ap.add_argument('--path_json', required=True)
    ap.add_argument('--out_json', required=True)
    ap.add_argument('--plot', action='store_true')
    args = ap.parse_args()
    
    annots,dt_category = get_annotations_from_json(args.path_json, args.path_imgs, args.plot)
    logger.info(f'found category: {dt_category}')
    
    dataset = COCO_Dataset(dt_category, args.path_imgs)
    dataset.add_annotations(annots, args.plot)
    dataset.write_to_json(args.out_json)
    