import os
import cv2
import numpy as np
import logging
from pathlib import Path
import time
import json
from anomalib_lmi.anomaly_model2 import AnomalyModel2

MAX_UINT16 = 65535
IMG_FORMATS = ['.png', '.jpg']
NUM_BINS = 100

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def predict(model_path, images_path, image_size, out_path, recursive=True, tile=None, stride=None, resize=False, overlap_mode='average', annotate=False):
    """generating anomaly maps for a set of images

    Args:
        model_path (str): the path to the anomaly detection model, supports .engine and .pt
        images_path (str): the path to input images
        image_size (list): the size of the input images (h,w)
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
    
    logger.info(f"Loading model: {model_path}.")
    model = AnomalyModel2(model_path, image_size=image_size, 
                          tile=tile, stride=stride, tile_mode='resize' if resize else 'padding')
    logger.info(f"Model loaded.")
    model.warmup()

    logger.info(f"Processing images")
    proctime = []
    anom_all,path_all = [],[]
    for idx, image_path in enumerate(images, 1):
        logger.info(f"Processing image [{idx}/{len(images)}]: {image_path}")
        image_path=str(image_path)
        img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        
        # inference
        t0 = time.time()
        anom_map = model.predict(img, **{"tiling_settings": {
            "overlap_mode": overlap_mode,
        }}).astype(np.float32)
        logger.debug(f'anom_map shape {anom_map.shape}')

        proctime.append(time.time() - t0)
        
        anom_all.append(anom_map)
        path_all.append(image_path)
    
    # Compute histogram
    logger.info(f"Computing anomaly score histogram for all data.")
    all_data_raveled = []
    for anom_map in anom_all:
        all_data_raveled.extend(np.squeeze(anom_map).ravel().tolist())

    data = np.array(all_data_raveled)
    global_min, global_max = data.min().item(), data.max().item()
    logger.debug(f'Global Min: {global_min}, Global Max: {global_max}')
    hist,bin_edges = np.histogram(data, bins=NUM_BINS, density=True)
    logger.debug(f"Anomaly score histogram: {hist}")
    logger.debug(f"Anomaly score bins: {bin_edges}")

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
        anom_map = anom.copy()
        anom = np.squeeze(anom)
        cur_max = anom.max()
        anom = (anom-global_min)/(global_max-global_min)*MAX_UINT16
        anom = anom.astype(np.uint16)
        
        # write anomaly map
        ext = os.path.splitext(path_src)[1]
        relpath = os.path.relpath(path_src, images_path)
        path_anom = os.path.join(out_path, relpath.replace(ext,'_anom.png'))
        if not os.path.exists(os.path.dirname(path_anom)):
            os.makedirs(os.path.dirname(path_anom))
        if annotate:
            annotated_image = model.annotate(cv2.imread(path_src), ad_scores=anom_map, ad_threshold=anom_map.min(), ad_max=anom_map.max())
            cv2.imwrite(os.path.join(out_path, relpath.replace(ext,'_anot.png')), annotated_image)

        logger.debug(f'write anomaly map to {path_anom}')
        cv2.imwrite(path_anom,anom)
        
        # append to out_dict
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
    ap.add_argument('--height',type=int, required=False, help='input height', default=224)
    ap.add_argument('--width',type=int, required=False, help='image width', default=224)
    ap.add_argument('--recursive', action='store_true', help='search images recursively')
    ap.add_argument('--tile', type=int, nargs='*', help='tile hight and width. Can be a single int or two integers')
    ap.add_argument('--stride', type=int, nargs='*', help='stride hight and width. Can be a single int or two integers')
    ap.add_argument('--resize', action='store_true', help='interpolate if it needs to resize images, otherwise pad zeros')
    ap.add_argument('--annotate', action='store_true', help='interpolate if it needs to resize images, otherwise pad zeros')
    ap.add_argument('--overlap_mode','-om', type=str, required=False,default="gaussian", help='overlap mode for tiling, can be "average", "max", "cosine", "linear", "gaussian"')
    args = ap.parse_args()
    if args.tile is not None:
        if len(args.tile) not in (1, 2):
            ap.error("--tile requires 1 or 2 integers")
        if len(args.stride) not in (1, 2):
            ap.error("--stride requires 1 or 2 integers")
        if len(args.tile) == 1:
            args.tile = [args.tile[0], args.tile[0]]
        if len(args.stride) == 1:
            args.stride = [args.stride[0], args.stride[0]]
    
    predict(args.model, args.images, [args.height,args.width] ,args.output, args.recursive, args.tile, args.stride, args.resize, args.overlap_mode, args.annotate)