from rf_detr_lmi.model import RFDETR
import glob
import os
import cv2
import numpy as np
import json
import logging

# setup the logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def find_images(path:str, exts=['jpg','jpeg','png']):
    """find all images with the given extensions in the path

    Args:
        path (str): the input path
        exts (list): the list of extensions

    Returns:
        list: the list of image paths
    """
    imgs = []
    for ext in exts:
        imgs.extend(glob.glob(os.path.join(path,f'*.{ext}')))
    return imgs


def inference_run(args):
    model_path = args.get('weights')
    imgs_path = args.get('input')
    out_path = args.get('output')

    if not os.path.exists(out_path):
        os.makedirs(args.output)
    
    # load model
    model = RFDETR(model_path)
    # model warmup
    model.warmup()
    # find images
    img_list = find_images(imgs_path)
    logger.info(f'Found {len(img_list)} images in {imgs_path}')

    for img_path in img_list:
        img_name = os.path.basename(img_path)
        logger.info(f'Processing image: {img_name}')
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        outputs = model.predict(image)

def main ():
    import argparse
    parser = argparse.ArgumentParser(description="RF-DETR-LMI Inference")
    parser.add_argument('--weights', type=str, required=True, help='Path to model weights')
    parser.add_argument('--input', type=str, required=True, help='Path to input images')
    parser.add_argument('--output', type=str, required=True, help='Path to save output results')
    args = parser.parse_args()
    
    inference_run(vars(args))

if __name__ == "__main__":
    main()




