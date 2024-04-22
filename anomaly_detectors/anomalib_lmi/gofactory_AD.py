import os
import cv2
import numpy as np
import logging
from pathlib import Path
import time
import json
from anomalib_lmi.anomaly_model import AnomalyModel

MAX_UINT16 = 65535
IMG_FORMATS = ['.png', '.jpg']

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



def predict(model_path, images_path, out_path, recursive=True):
    """generating anomaly maps for a set of images

    Args:
        model_path (str): the path to the anomaly detection model, supports .engine and .pt
        images_path (str): the path to input images
        out_path (str): the output path to save the anomaly maps and summary.json
        recursive (bool, optional): whether to search images recursively. Defaults to True.
    """

    directory_path=Path(images_path)
    images = []
    for fm in IMG_FORMATS:
        if recursive:
            images.extend(directory_path.rglob(f'*{fm}'))
        else:
            images.extend(directory_path.glob(f'*{fm}'))
    logger.info(f"{len(images)} images from {images_path}")
    if not images:
        return
    
    logger.info(f"Loading engine: {model_path}.")
    pc = AnomalyModel(model_path)
    
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    pc.warmup()

    proctime = []
    anom_all,path_all = [],[]
    for image_path in images:
        logger.info(f"Processing image: {image_path}.")
        image_path=str(image_path)
        img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        
        # inference
        t0 = time.time()
        anom_map = pc.predict(img).astype(np.float32)
        proctime.append(time.time() - t0)
        
        anom_all.append(anom_map)
        path_all.append(image_path)
    
    # Compute histogram
    logger.info(f"Computing anomaly score histogram for all data.")
    anom_sq=np.squeeze(np.array(anom_all))
    data=np.ravel(anom_sq)
    global_min,global_max = data.min().item(),data.max().item()
    bins = [i for i in range(int(global_min),int(global_max)+1)]
    hist,bin_edges = np.histogram(data, bins=bins, density=True)
    logger.info(f"Anomaly score histogram: {hist}")
    logger.info(f"Anomaly score bins: {bin_edges}")
    
    out_dict = {
        'summary':{
            'anomaly_distribution':{},
            'anomaly_max': global_max,
            'anomaly_min': global_min,
            },
        'images':[],
    }
    out_dict['summary']['anomaly_distribution']['anomaly_score'] = bin_edges.tolist()
    out_dict['summary']['anomaly_distribution']['probability'] = hist.tolist()
    
    for path_src,anom in zip(path_all,anom_all):
        # normalize to uint16
        anom = np.squeeze(anom)
        cur_max = anom.max()
        anom = (anom-global_min)/(global_max-global_min)*MAX_UINT16
        anom = anom.astype(np.uint16)
        fname,_ = os.path.splitext(os.path.basename(path_src))
        path_anom = os.path.join(out_path,fname+'_anom.png')
        logger.info(f'write anomaly map to {path_anom}')
        cv2.imwrite(path_anom,anom)
        cur = {
            'source_image_path': os.path.relpath(path_src, images_path),
            'anomaly_image_path': os.path.relpath(path_anom, out_path),
            'anomaly_max': cur_max.item(),
        }
        out_dict['images'].append(cur)
    
    # write to json
    path_json = os.path.join(out_path, 'summary.json')
    with open(path_json,'w') as f:
        json_str = json.dumps(out_dict)
        f.write(json_str)
    
    if len(proctime):
        proctime = np.asarray(proctime)
        logger.info(f'Min Proc Time: {proctime.min()}')
        logger.info(f'Max Proc Time: {proctime.max()}')
        logger.info(f'Avg Proc Time: {proctime.mean()}')
        logger.info(f'Median Proc Time: {np.median(proctime)}')
    logger.info(f"Test results saved to {out_path}")
    
    
    
if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='gofactory AD prediction')
    ap.add_argument('--model', type=str, required=True, help='path to the AD model')
    ap.add_argument('-i','--images', type=str, required=True, help='path to the testing images')
    ap.add_argument('-o','--output', type=str, required=True, help='path to the output folder')
    ap.add_argument('--recursive', action='store_true', help='search images recursively')
    args = ap.parse_args()
    
    predict(args.model, args.images, args.output, args.recursive)