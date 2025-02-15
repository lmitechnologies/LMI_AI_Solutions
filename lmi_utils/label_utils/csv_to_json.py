import csv
import os
import cv2
import pathlib

from dataset_utils.representations import Box, Mask, Label, Annotation, File, AnnotationType, Dataset, FileAnnotations
from dataset_utils.mask_encoder import mask2rle, rle2mask


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
    c1 = list(map(float, coordinates))
    c2 = list(map(float, coordinates2))
    return fname, category, conf, shape_type, c1, c2


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
        
        label_map = {}
        file_map = {}
        annot_id = 0
        for i in range(0,len(reader),2):
            row1 = reader[i]
            row2 = reader[i+1]
            fname,category,conf,shape_type,c1,c2 = read_two_rows(row1, row2)
            
            if shape_type == 'polygon':
                xy = [[x,y] for x,y in zip(c1,c2)]
                shape = Mask(MaskType.POLYGON,xy)
                mtype = AnnotationType.MASK
            if shape_type == 'rect':
                if len(c1)==4:
                    angle = float(c1[-1])
                shape = Box(x_min=c1[0], y_min=c1[1], x_max=c2[0], y_max=c2[1], angle=angle)
                mtype = AnnotationType.BOX
            if shape_type == 'keypoint':
                shape = Point(x=c1[0], y=c1[1], z=0)
                mtype = AnnotationType.POINT
                
            if category not in label_map:
                label_id = len(label_map)
                label_map[category] = label_id
            else:
                label_id = label_map[category]
                
            if fname not in file_map:
                p = os.path.join(img_dir, fname)
                im = cv2.imread(p)
                height, width = im.shape[:2]
                # TODO: add id to filename
                file_id = len(file_map)
                file = File(id=str(file_id), path=fname, height=height, width=width)
                file_map[fname] = FileAnnotations(file=file, annotations=[], predictions=[])
            if fname in file_map:
                file = file_map[fname]
                annot = Annotation(id=str(annot_id),label_id=category,type=mtype,value=shape,confidence=conf,link=Link())
                annot_id += 1
                file.annotations.append(annot)
            
    return label_map, file_map


def write_to_json(label_map:dict, file_map:dict, json_path:str):
    """write to a json file using the new representation classes

    Args:
        label_map (dict): a dictionary mapping label names to label ids
        file_map (dict): a dictionary mapping filenames to FileAnnotations
        json_path (str): path to a json file
    """
    dataset = Dataset(labels=[], files=[])
    dataset.labels = [Label(index=str(label_id), id=label_name) for label_name,label_id in label_map.items()]
    dataset.files = list(file_map.values())
    dataset.save(json_path)
    
    
if __name__ == '__main__':
    csv_path = 'data/tuber/labels.csv'
    img_dir = 'data/tuber'
    json_path = 'data/tuber/labels.json'
    label_map, file_map = read_csv(csv_path, img_dir)
    write_to_json(label_map, file_map, json_path)
    