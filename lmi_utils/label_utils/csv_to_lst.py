import os
import json
import cv2
import logging
import lxml.etree as ET

from label_utils.csv_utils import load_csv
from label_utils.shapes import Rect, Mask


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


OUT_NAME = 'lst.json'
RECT_NAME = 'label'
POLYGON_NAME = 'polygon'


def rect_to_lst(rect_obj, width, height, is_pred):
    x1,y1 = rect_obj.up_left
    x2,y2 = rect_obj.bottom_right
    w,h = x2-x1, y2-y1
    box = {
        'original_width': width,
        'original_height': height,
        'image_rotation': 0,
        'value': {
            'x': x1 / width * 100,
            'y': y1 / height * 100,
            'width': w / width * 100,
            'height': h / height * 100,
            'rotation': rect_obj.angle,
            'rectanglelabels': [rect_obj.category]
        },
        'from_name': RECT_NAME,
        'to_name': 'image',
        'type': 'rectanglelabels'
    }
    if is_pred:
        box['value']['score'] = rect_obj.confidence
    return box


def mask_to_lst(mask_obj, width, height, is_pred):
    X,Y = mask_obj.X, mask_obj.Y
    polygon = {
            "original_width": width,
            "original_height": height,
            "image_rotation": 0,
            'value': {
                'points': [],
                'polygonlabels': [mask_obj.category]
            },
            'from_name': POLYGON_NAME,
            'to_name': 'image',
            'type': 'polygonlabels'
        }
    for i in range(0, len(X)):
        x = X[i] / width * 100
        y = Y[i] / height * 100
        polygon['value']['points'].append([x,y])
    if is_pred:
        polygon['value']['score'] = mask_obj.confidence
    return polygon


def init_label_obj(path_img, is_pred):
    # TODO: support BOTH annotations and predictions
    label_obj = {}
    if not is_pred:
        label_obj['annotations'] = [
            {
                'result': []
            }
        ]
    else:
        label_obj['predictions'] = [
            {
                'model_version': 'prediction',
                'result': []
            }
        ]
    label_obj['data'] = {
        'image':path_img
    }
    return label_obj


def write_xml(out_path, box_class, polygon_class):
    root = ET.Element("View")
    image = ET.SubElement(root, "Image")
    image.set("name", "image")
    image.set("value", "$image")
    image.set("zoom", "true")
    
    if len(box_class) > 0:
        rect = ET.SubElement(root, "RectangleLabels")
        rect.set("name", RECT_NAME)
        rect.set("toName", "image")
        for name in box_class:
            label = ET.SubElement(rect, "Label")
            label.set("value", name)
            # label.set("background", "green")
    
    if len(polygon_class) > 0:
        polygon = ET.SubElement(root, "PolygonLabels")
        polygon.set("name", POLYGON_NAME)
        polygon.set("toName", "image")
        polygon.set("strokeWidth", "3")
        polygon.set("pointSize", "small")
        polygon.set("opacity", "0.9")
        for name in polygon_class:
            label = ET.SubElement(polygon, "Label")
            label.set("value", name)
            # label.set("background", "blue")
    
    str = ET.tostring(root, pretty_print=True, encoding='unicode')
    html_path = f"{out_path}/labeling_interface.xml"
    with open(html_path, 'w') as file:
        file.write(str)


def write_to_lst(shapes, out_path, images_path, gs_path, width, height, is_pred):
    if not gs_path.startswith('gs://'):
        if not gs_path.startswith('/'):
            raise Exception('The local storage path must be absolute path starting with /')
        l = gs_path.split('/')[1:]
        logger.info(f'found local path: {gs_path}.')
        logger.info(f'Assume that LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT is /{l[0]}')
        gs_path = f"/data/local-files/?d={str('/').join(l[1:])}"
        logger.info(f'converted to local path: {gs_path}')
        
    labels = []
    box_class = set()
    polygon_class = set()
    cnt_box = 0
    cnt_polygon = 0
    cnt_img = len(shapes)
    # sort the shapes by filename
    shapes = {k: v for k, v in sorted(shapes.items(), key=lambda item: item[0])}
    for fname in shapes:
        if images_path is not None:
            im = cv2.imread(os.path.join(images_path, fname))
            height,width = im.shape[:2]

        label_obj = init_label_obj(os.path.join(gs_path, fname), is_pred)
        target = 'predictions' if is_pred else 'annotations'
        
        for shape in shapes[fname]:
            if isinstance(shape, Rect):
                box = rect_to_lst(shape, width, height, is_pred)
                label_obj[target][0]['result'].append(box)
                cnt_box += 1
                box_class.add(shape.category)
            elif isinstance(shape, Mask):
                polygon = mask_to_lst(shape, width, height, is_pred)
                label_obj[target][0]['result'].append(polygon)
                cnt_polygon += 1
                polygon_class.add(shape.category)
            else:
                raise Exception(f'Invalid shape type: {type(shape)}')
        labels.append(label_obj)

    # save to json
    with open(os.path.join(out_path,OUT_NAME), 'w') as f:
        json.dump(labels, f, indent=4)
        
    # write the xml file
    write_xml(out_path, box_class, polygon_class)
    
    logger.info(f'Number of images: {cnt_img}')
    logger.info(f'Number of boxes: {cnt_box}')
    logger.info(f'Number of polygons: {cnt_polygon}')
    logger.info(f'In the label studio labeling interface, ensure that PolygonLabels name="{POLYGON_NAME}" and RectangleLabels name="{RECT_NAME}"')


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='This script requires either --path_imgs or --wh, but not both.')
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument('--path_imgs', help='path to the images, where images have different dimensions')
    group.add_argument('--wh', help='width and height of the images separated by a comma, if they are the same dimension')
    ap.add_argument('--csv', required=True, help='path to the csv file')
    ap.add_argument('--lst_img_dir', required=True, help='the image directory will be output in the label studio json. Either a gs path (start with gs://) or local absolute path (start with /)')
    ap.add_argument('--out_dir', '-o', required=True, help='the output directory')
    ap.add_argument('--pred', action='store_true', help='if the csv file is a prediction file')
    args = ap.parse_args()
    
    width,height = None,None
    if args.wh is not None:
        wh = args.wh.split(',')
        width = int(wh[0])
        height = int(wh[1])

    if os.path.isfile(args.out_dir):
        raise Exception('The output path should be a directory')
    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)

    shapes = load_csv(args.csv)[0]
    write_to_lst(shapes, args.out_dir, args.path_imgs, args.lst_img_dir, width, height, args.pred)
