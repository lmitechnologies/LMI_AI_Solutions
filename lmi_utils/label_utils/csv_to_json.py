import csv
import os
import cv2
import numpy as np

from dataset_utils.representations import Box, Mask, Polygon, Point2d, Label, Annotation, AnnotationType, Dataset, FileAnnotations


def read_one_row(row):
    im_name = row[0]
    category = row[1]
    try:
        # expect to find confidence level
        confidence = float(row[2])
        shape_type = row[3]
        coord_type = row[4]
        coordinates = row[5:]
    except Exception:
        # incase cannot find confidence level, set it to 1.0
        confidence = 1.0
        shape_type = row[2]
        coord_type = row[3]
        coordinates = row[4:]
    return im_name, category, confidence, shape_type, coord_type, coordinates


def read_two_rows(row1, row2):
    fname, category, conf, shape_type, coord_type, coordinates = read_one_row(row1)
    fname2, category2, conf2, shape_type2, coord_type2, coordinates2 = read_one_row(row2)
    
    assert fname == fname2 and category == category2 and conf == conf2 and shape_type == shape_type2
    
    angle = None
    if shape_type == 'rect':
        if len(coordinates)==4:
            angle = float(coordinates[-1])
            coordinates = coordinates[:-2]
            coordinates2 = coordinates2[:-2]
    c1 = list(map(float, coordinates))
    c2 = list(map(float, coordinates2))
    return fname, category, conf, shape_type, c1, c2, angle


def read_csv(csv_path:str, img_dir:str):
    """read csv using the new representation classes

    Args:
        csv_path (str): path to a csv file
        img_dir (str): a directory containing images

    Returns:
        list: a list of two maps
    """
    # TODO: load predictions
    with open(csv_path, newline='') as f:
        reader = list(csv.reader(f, delimiter=';'))
        
        label_set = set()
        file_map = {}
        annot_id = 0
        for i in range(0,len(reader),2):
            row1 = reader[i]
            row2 = reader[i+1]
            fname,category,conf,shape_type,c1,c2,angle = read_two_rows(row1, row2)
            p = os.path.join(img_dir, fname)
            im = cv2.imread(p)
            height, width = im.shape[:2]
            
            if shape_type == 'polygon':
                xy = np.array([[x,y] for x,y in zip(c1,c2)]).astype(int)
                # mask = np.zeros((height, width), dtype=np.uint8)
                # cv2.fillPoly(mask, [xy], 1)
                # shape = Mask(mask)
                # mtype = AnnotationType.MASK
                shape = Polygon(xy.tolist())
                mtype = AnnotationType.POLYGON
            if shape_type == 'rect':
                shape = Box(x_min=c1[0], y_min=c1[1], x_max=c2[0], y_max=c2[1], angle=angle)
                mtype = AnnotationType.BOX
            if shape_type == 'keypoint':
                shape = Point2d(x=c1[0], y=c2[0])
                mtype = AnnotationType.KEYPOINT
                
            label_set.add(category)
                
            if fname not in file_map:
                # TODO: add id to filename
                file_id = len(file_map)
                file_map[fname] = FileAnnotations(id=str(file_id), path=fname, height=height, width=width, annotations=[], predictions=[])
            if fname in file_map:
                file = file_map[fname]
                annot = Annotation(id=str(annot_id),label_id=str(category),type=mtype,value=shape,confidence=conf)
                annot_id += 1
                file.annotations.append(annot)
            
    return label_set, file_map


def write_to_json(label_set:dict, file_map:dict, json_path:str):
    """write to a json file using the new representation classes

    Args:
        label_set (dict): a set of labels
        file_map (dict): a dictionary mapping filenames to FileAnnotations
        json_path (str): path to a json file
    """
    dataset = Dataset(labels=[], files=[])
    dataset.labels = [Label(id=str(label_name)) for label_name in label_set]
    dataset.files = list(file_map.values())
    dataset.save(json_path)
    
    
if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv',required=True)
    ap.add_argument('-i','--img_dir',required=True)
    ap.add_argument('-o','--json_path',required=True,help='a output json path')
    args = ap.parse_args()
    
    label_map, file_map = read_csv(args.csv, args.img_dir)
    write_to_json(label_map, file_map, args.json_path)
    