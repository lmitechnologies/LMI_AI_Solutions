import os
import argparse
import logging
import json
import numpy as np
import collections
import glob
from label_studio_sdk.converter.brush import decode_rle
from dataset_utils.representations import Box, Mask, Label, AnnotationType, Dataset, FileAnnotations, Polygon, Point2d, Annotation
from dataset_utils.mask_encoder import mask2rle, rle2mask
from system_utils.path_utils import get_relative_paths
import cv2
import shutil


from label_utils.bbox_utils import convert_from_ls

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

LABEL_NAME = 'labels.json'
PRED_NAME = 'preds.json'
IMAGES_DIR = 'images'



def lst_to_shape(result:dict, fname:str, load_confidence=False):
    """parse the result from label studio result dict, return a Shape object
    """
    result_type=result['type']
    labels = result['value'][result_type]
    if len(labels) > 1:
        raise Exception('Not support more than one labels in a bbox/polygon')
    if len(labels) == 0:
        logger.warning(f'found empty label in {fname}, skip')
        return None, None, None, None
    
    label = labels[0]
    conf = result['value']['score'] if load_confidence else 1.0
    if result_type=='rectanglelabels':   
        # get bbox
        x,y,w,h,angle = convert_from_ls(result)
        x1,y1,w,h = list(map(int,[x,y,w,h]))
        x2,y2 = x1+w, y1+h
        box = Box(x_min=x1,y_min=y1,x_max=x2,y_max=y2,angle=angle)
        return box, label, conf, AnnotationType.BOX
    elif result_type=='polygonlabels':
        points=result['value']['points']
        points_np=np.array(points)
        xs = (points_np[:, 0]/100*result['original_width']).astype(np.int32)
        ys = (points_np[:, 1]/100*result['original_height']).astype(np.int32)
        points_np = np.stack([xs,ys],axis=1)
        return Polygon(points=points_np.astype(int).tolist()), label, conf, AnnotationType.POLYGON
    elif result_type=='brushlabels':
        rle = result['value']['rle']
        h,w = result['original_height'],result['original_width']
        img = decode_rle(rle).reshape(h,w,4)[:,:,3]
        mask = img > 128
        return Mask(mask=mask), label, conf, AnnotationType.MASK
    elif result_type=='keypointlabels':
        dt = result['value']
        x,y = dt['x']/100*result['original_width'],dt['y']/100*result['original_height']
        return Point2d(x=x, y=y), label, conf, AnnotationType.KEYPOINT
    else:
        logger.warning(f'unsupported result type: {result_type}, skip')

def generate_file_ids(files):
    file_id = {}
    for i, f in enumerate(files):
        file_id[f] = i
    return file_id
        

def get_annotations_from_json(path_json, images_dir, background=False):
    """read annotation from label studio json file.

    Args:
        path_json (str): the path to a directory of label studio json files

    Returns:
        dict: a map <image name, a list of Rect objects>
    """
    if os.path.splitext(path_json)[1]=='.json':
        json_files=[path_json]
    else:
        json_files=glob.glob(os.path.join(path_json,'*.json'))

    labels : list[Label] = []
    annotations: list[FileAnnotations] = []
    
    label_dict = {}
    file_id_dict = generate_file_ids(get_relative_paths(images_dir))
    
    for path_json in json_files:
        if path_json.endswith(LABEL_NAME) or path_json.endswith(PRED_NAME):
            continue
        logger.info(f'Extracting labels from: {path_json}')
        logger.info(f'dir_path : {images_dir}')
        with open(path_json) as f:    
            l = json.load(f)

        cnt_anno = 0
        cnt_image = 0
        cnt_pred = 0
        cnt_wrong = 0
        
        # collect all the files
        files = [
            dt['data']['image'] for dt in l if 'data' in dt
        ]
        common_prefix = os.path.dirname(os.path.commonprefix(files))
        logger.info(f'base_path: {common_prefix}')
        
        # find the common prefix between the image path
        
        
        for dt in l:
            # load file name
            if 'data' not in dt:
                raise Exception('missing "data" in json file. Ensure that the label studio export format is not JSON-MIN.')
            f = dt['data']['image'] # image web path
            file_annotations : list[Annotation] = []
            pred_annotations : list[Annotation] = []
            
            if 'annotations' in dt:
                cnt = 0
                for annot in dt['annotations']:
                    num_labels = len(annot['result'])
                    if num_labels>0:
                        cnt += 1
                    for result in annot['result']:
                        shape, label, conf, annot_type = lst_to_shape(result,f)
                        if shape is not None:
                            if label not in label_dict:
                                label_id = label
                                label_dict[label] = label_id
                                labels.append(Label(id=str(label_id), annotation_type=annot_type))
                            else:
                                label_id = label_dict[label]  
                            file_annotations.append(Annotation(id=str(cnt_anno), label_id=str(label_id), type=annot_type, value=shape))
                            cnt_anno += 1
                            
                            
                            
                    if 'prediction' in annot and 'result' in annot['prediction']:
                        for result in annot['prediction']['result']:
                            shape, label, conf, annot_type = lst_to_shape(result,f,load_confidence=True)
                            if shape is not None:
                                if label not in label_dict:
                                    label_id = label
                                    label_dict[label] = label_id
                                    labels.append(Label(id=str(label_id), annotation_type=annot_type))
                                else:
                                    label_id = label_dict[label]
                                pred_annotations.append(Annotation(id=str(cnt_pred), label_id=str(label_id), type=annot_type, value=shape, confidence=conf))
                                cnt_pred += 1
                if cnt>0:
                    cnt_image += 1
                if cnt==0 and dt['total_annotations']>0:
                    cnt_wrong += 1
                    logger.warning(f'found 0 annotation in {f}, but lst claims total_annotations = {dt["total_annotations"]}')

            if 'predictions' in dt:
                for pred in dt['predictions']:
                    if isinstance(pred, dict):
                        for result in pred['result']:
                            shape, label, conf, annot_type = lst_to_shape(result,f,load_confidence=True)
                            if shape is not None:
                                if label not in label_dict:
                                    label_id = label
                                    label_dict[label] = label_id
                                    labels.append(Label(id=str(label_id), annotation_type=annot_type))
                                else:
                                    label_id = label_dict[label]
                                pred_annotations.append(Annotation(id=str(cnt_pred), label_id=str(label_id), type=annot_type, value=shape, confidence=conf))
                                cnt_pred += 1
            
            f = f.removeprefix(common_prefix).removeprefix('/')
            updated_fp = os.path.join(images_dir, f)
            if not os.path.isfile(updated_fp):
                raise Exception(f'file not found: {updated_fp}')
            
            file_id = file_id_dict[f]
                
                
            image = cv2.imread(updated_fp, cv2.IMREAD_UNCHANGED)
            height, width = image.shape[:2]
            if len(file_annotations)>0:
                # File(id=str(file_id), path=f, height=height, width=width)
                annotations.append(FileAnnotations(id=file_id,path=f, height=height,width=width, annotations=file_annotations, predictions=pred_annotations))
                cnt_image += 1

            else:
                logger.warning(f'no annotation found in {f}')
                
                if background:
                    annotations.append(FileAnnotations(id=str(file_id),path=f, height=height,width=width))
                

        logger.info(f'{cnt_image} out of {len(l)} images have annotations')
        if cnt_wrong>0:
            logger.info(f'{cnt_wrong} images with total_annotations > 0, but found 0 annotation')
        logger.info(f'total {cnt_anno} annotations')
        logger.info(f'total {cnt_pred} predictions')
    # save all background images
    if background:
        for f in file_id_dict:
            updated_fp = os.path.join(images_dir, f)
            if not os.path.isfile(updated_fp):
                raise Exception(f'file not found: {updated_fp}')
            file_id = file_id_dict[f]
            image = cv2.imread(updated_fp, cv2.IMREAD_UNCHANGED)
            height, width = image.shape[:2]
            annotations.append(FileAnnotations(id=str(file_id),path=f, height=height,width=width))
    
    logger.info(f'total {len(annotations)} images')
    logger.info(f'total {len(labels)} labels')
    
    
    return annotations, labels

def main():
    ap = argparse.ArgumentParser('Convert label studio json file to json format')
    ap.add_argument('-i', '--path_json', required=True, help='the directory of label-studio json files')
    ap.add_argument('-imgs', '--path_images', required=False, help='the root directory of images')
    ap.add_argument('-of', '--path_out_json', required=False, help='path to store the json file')
    ap.add_argument('-bg', '--background', action='store_true', help='save background')
    args = ap.parse_args()
    
    
    
    annotations, labels = get_annotations_from_json(args.path_json, args.path_images, background=args.background)
    
    annotations = Dataset(labels=labels, files=annotations)
    out_path = args.path_out_json
    if not out_path.endswith('.json') and out_path!='labels.json':
        if not os.path.isdir(out_path):
            os.makedirs(out_path)
        out_json = os.path.join(out_path, 'labels.json')
    else:
        out_json = out_path
    
    annotations.save(out_json)
    logger.info(f'saved to {out_json}')
    

if __name__ == '__main__':
    main()

    
