import torch
import os

# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import cv2, random

# import some common detectron2 utilities
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances


def plot(path_img, path_json, path_out, plot=False):
    """
    plot to verify if dataset is loaded correctly
    arguments:
        path_img(str): the path to the image folder
        path_json(str): the path to the json annotation file
    """
    register_coco_instances("dataset", {}, path_json, path_img)
    dataset_dicts = DatasetCatalog.get("dataset")
    for d in dataset_dicts:
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=MetadataCatalog.get('dataset'), scale=1)
        out = visualizer.draw_dataset_dict(d)
        im_name = 'annot_'+os.path.basename(d["file_name"])
        out.save(os.path.join(path_out, im_name))
        if plot:
            cv2.imshow('plot',out.get_image()[:, :, ::-1])
            cv2.waitKey(0)
    if plot:
        cv2.destroyAllWindows()

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--path_img', required=True, type=str, help='the path to the image folder')
    ap.add_argument('--path_json', type=str, default='', help='[optinal] the path to the json annotation file, default=labels.json')
    ap.add_argument('--path_out', required=True, type=str, help='the path to the output folder')
    args = ap.parse_args()

    TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
    CUDA_VERSION = torch.__version__.split("+")[-1]
    print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)
    
    if not args.path_json:
        args.path_json = os.path.join(args.path_img, 'labels.json')
    if not os.path.isdir(args.path_out):
        os.makedirs(args.path_out)

    plot(args.path_img, args.path_json, args.path_out)