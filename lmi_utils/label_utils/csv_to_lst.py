import os
import json
import cv2
import logging

from label_utils.csv_utils import load_csv
from label_utils import rect
from label_utils import mask


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
    labeling_interface = "<View>\n"
    labeling_interface += "  <Header value=\"Select label and click the image to start\"/>\n"
    labeling_interface += "  <Image name=\"image\" value=\"$image\" zoom=\"true\"/>\n"
    
    if len(box_class) > 0:
        labeling_interface += f"  <RectangleLabels name=\"{RECT_NAME}\" toName=\"image\">\n"
        for name in box_class:
            labeling_interface += f"    <Label value=\"{name}\" background=\"green\"/>\n"
        labeling_interface += "  </RectangleLabels>"
    
    if len(polygon_class) > 0:
        labeling_interface += f"  <PolygonLabels name=\"{POLYGON_NAME}\" toName=\"image\" strokeWidth=\"3\" pointSize=\"small\" opacity=\"0.9\">\n"
        for name in polygon_class:
            labeling_interface += f"    <Label value=\"{name}\" background=\"blue\"/>\n"
        labeling_interface += "  </PolygonLabels>\n"
    
    labeling_interface += "</View>"

    html_path = f"{out_path}/labeling_interface.xml"
    with open(html_path, 'w') as file:
        file.write(labeling_interface)


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
    for fname in shapes:
        if images_path is not None:
            im = cv2.imread(os.path.join(images_path, fname))
            height,width = im.shape[:2]

        label_obj = init_label_obj(os.path.join(gs_path, fname), is_pred)
        target = 'predictions' if is_pred else 'annotations'
        
        for shape in shapes[fname]:
            if isinstance(shape, rect.Rect):
                box = rect_to_lst(shape, width, height, is_pred)
                label_obj[target][0]['result'].append(box)
                cnt_box += 1
                box_class.add(shape.category)
            elif isinstance(shape, mask.Mask):
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
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True, help='path to the csv file')
    ap.add_argument('--path_imgs', help='[optional] path to the images')
    ap.add_argument('--wh', help='[optional] width and height of the images separated by a comma. Assume that images are the same size')
    ap.add_argument('--path_gs', required=True, help='the gs path or local absolute path')
    ap.add_argument('--path_out', '-o', required=True)
    ap.add_argument('--pred', action='store_true', help='if the csv file is a prediction file')
    args = ap.parse_args()

    if args.path_imgs is None and args.wh is None:
        raise Exception('Provide the path to the images. Or if the images are the same size, provide the width and height')
    if args.wh is not None:
        wh = args.wh.split(',')
        width = int(wh[0])
        height = int(wh[1])

    if os.path.isfile(args.path_out):
        raise Exception('The output path should be a directory')
    if not os.path.isdir(args.path_out):
        os.makedirs(args.path_out)

    shapes = load_csv(args.csv)[0]
    write_to_lst(shapes, args.path_out, args.path_imgs, args.path_gs, width, height, args.pred)