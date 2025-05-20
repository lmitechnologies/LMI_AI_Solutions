import numpy as np
import cv2
import os
import logging

#LMI packages
from dataset_utils.representations import Dataset,AnnotationType
from label_utils.plot_utils import plot_one_polygon, plot_one_pt, plot_one_brush
from label_utils.bbox_utils import rotate
from label_utils.plot_utils import get_distinct_colors

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def plot_shape(shape, im, color_map, no_label=False):
    img_h, img_w = im.shape[:2]
    label = shape.label_id
    if label not in color_map:
        color_map[label] = [255,255,255]
    label_str = None if no_label else label
    if shape.type == AnnotationType.BOX:
        x1,y1, x2, y2, angle = shape.value.coords()
        width = x2 - x1
        height = y2 - y1
        # rotated rectangle
        if angle > 0:
            rotated_rect = rotate(x1, y1, width, height, angle)
        else:
            rotated_rect = np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]])
        plot_one_polygon(np.array([rotated_rect]), im, label=label_str, color=color_map[label])
    elif shape.type == AnnotationType.POLYGON:
        pts = shape.value.to_numpy().reshape((-1, 1, 2)).astype(int)
        plot_one_polygon(pts, im, label=label_str, color=color_map[label])
    elif shape.type == AnnotationType.MASK:
        x,y = shape.value.coords(h=img_h, w=img_w)
        plot_one_brush(x,y,im,label=label_str,color=color_map[label])
    elif shape.type == AnnotationType.KEYPOINT:
        x,y, = shape.value.coords()
        plot_one_pt([x,y], im, label=label_str, color=color_map[label])
    else:
        raise Exception(f'Unknown shape: {type(shape)}')
    return



if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('-i','--path_imgs', required=True, help='the path to the input image folder')
    ap.add_argument('-o','--path_out', required=True, help='the path to the output folder')
    ap.add_argument('--path_json', default='labels.json', help='[optinal] the path of a csv file that corresponds to path_imgs, default="labels.json" in path_imgs')
    ap.add_argument('--preds', action='store_true', help='[optional] plot predictions on the image')
    ap.add_argument('--no_label', action='store_true', help='[optional] do not show label on the image')
    args = vars(ap.parse_args())

    path_imgs = args['path_imgs']
    path_json = args['path_json'] if args['path_json']!='labels.json' else os.path.join(path_imgs, args['path_json'])
    output_path = args['path_out']
    assert path_imgs!=output_path, 'output path must be different with input path'
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    # fname_to_shape, class_map = load_csv(path_json, path_imgs, class_map)
    
    dataset = Dataset.load(path_json)
    base_prefix = dataset.base_path
    
    # init color map
    color_map = {}
    colors = get_distinct_colors(len(dataset.get_label_ids()))
    for i,name in enumerate(dataset.get_label_ids()):
        logger.info(f'CLASS: {name}')
        color_map[name] = tuple(colors[i])
    
    for f in dataset.files:
        file_path = os.path.join(path_imgs, f.path)
        
        fname = os.path.basename(file_path)
        logger.info(f'processing {fname}')
        if not os.path.isfile(file_path) and f.has_annotations:
            logger.warning(f'file not found: {file_path} has annotations {f.annotations}')
            raise Exception(f'file not found: {file_path}')
        elif not os.path.exists(file_path):
            logger.warning(f'file not found: {file_path}')
            continue
        im0 = cv2.imread(file_path)
        
        im_annot = im0.copy()
        for shape in f.annotations:
            plot_shape(shape, im_annot, color_map, args['no_label'])
            
        if f'id{f.id}_' not in fname:
            fname = f'id{f.id}_{fname}'

        #create output fname and save it
        out_name = os.path.splitext(fname)[0] + f'_annot' + '.png'
        output_file=os.path.join(output_path, out_name)
        cv2.imwrite(output_file, im_annot)
        
        if args['preds'] and len(f.predictions):
            im_pred = im0.copy()
            for shape in f.predictions:
                plot_shape(shape, im_pred, color_map, args['no_label'])
            
            root,ext = os.path.splitext(fname)

            #create output fname and save it
            out_name = os.path.splitext(fname)[0] + f'_pred' + '.png'
            output_file=os.path.join(output_path, out_name)
            cv2.imwrite(output_file, im_pred)
            
