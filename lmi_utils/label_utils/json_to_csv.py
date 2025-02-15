from label_utils import csv_utils
from label_utils.shapes import Rect, Mask, Keypoint, Brush
from system_utils.path_utils import get_relative_paths
from dataset_utils.representations import Dataset, AnnotationType
import collections
from label_utils.csv_utils import write_to_csv
import os


def json_to_csv(path_json, path_out):
    # load dataset
    dataset = Dataset.load(path_json)
    base_prefix = dataset.base_path
    fname_to_shapes = collections.defaultdict(list)
    
    for f in dataset.files:
        relative_path = f.relative_path(base_prefix)
        fname = os.path.basename(relative_path)
        image_h = f.height
        image_w = f.width
        for annotation in f.annotations:
            conf = annotation.confidence
            label = dataset.label_id_to_name(annotation.label_id)
            if conf is None:
                conf = 1.0
            if annotation.type == AnnotationType.BOX:
                # rect
                x1,y1,x2,y2, angle = annotation.value.coords()
                fname_to_shapes[fname].append(Rect(im_name=fname,category=label,up_left=[x1,y1],bottom_right=[x2,y2],angle=angle,confidence=conf))
            elif annotation.type == AnnotationType.MASK:
                mask = annotation.value.to_numpy(h=image_h, w=image_w)
                fname_to_shapes[fname].append(Brush(im_name=fname,category=label,mask=mask,confidence=conf))
            
            elif annotation.type == AnnotationType.POLYGON:
                xs, ys = annotation.type.coords()
                fname_to_shapes[fname].append(Mask(im_name=fname,category=label,x_vals=xs,y_vals=ys,confidence=conf))
            elif annotation.type == AnnotationType.KEYPOINT:
                x,y = annotation.value.x, annotation.value.y
                fname_to_shapes[fname].append(Keypoint(im_name=fname,category=label,x=x,y=y,confidence=conf))
    
    json_file_name = os.path.basename(path_json)
    csv_file_name = json_file_name.replace('.json', '.csv')
    csv_file_out = os.path.join(path_out, csv_file_name)
    write_to_csv(fname_to_shapes, csv_file_out)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_json', type=str, help='path to json file')
    parser.add_argument('--path_out', type=str, help='path to output csv file')
    args = parser.parse_args()
    json_to_csv(args.path_json, args.path_out)
    
    
                          
        
