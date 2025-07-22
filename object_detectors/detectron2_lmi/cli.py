import argparse
from detectron2_lmi.trainer import training_run
from detectron2_lmi.infer import inference_run
from detectron2_lmi.convert import convert

import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


DET2_ONNX_EXPORT = 'model.onnx'
DET2_TRT_EXPORT = 'model.engine'
DET2_PT_EXPORT = 'model.pt'
DET2_DEFAULT_DIR = '/home/weights'
DET2_DATASET_DIR = '/home/data'
DET2_INPUT_DIR = '/home/input'
DET2_OUTPUT_DIR = '/home/output'
DET2_CLASS_MAP = '/home/class_map.json'
DET2_CONFIG_FILE = 'config.yaml'
DET2_SAMPLE_IMAGE = '/home/sample_image.png'
DET2_PTH_EXPORT = 'model_final.pth'

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    subs = ap.add_subparsers(dest='action',required=True,help='Action modes: train, test or convert')
    
    train_ap = subs.add_parser('train',help='train model')
    train_ap.add_argument('-c',"--config-file",metavar="FILE", help="path to config file", default=os.path.join('/home', DET2_CONFIG_FILE))
    train_ap.add_argument("--detectron2-config", f'-det2config',type=str, help="Detectron2 config file", default="COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    train_ap.add_argument("--dataset_dir", type=str, help="Dataset dir that contains the train and validation datasets", default=DET2_DATASET_DIR)
    train_ap.add_argument("--output", type=str, help="Path to the output directory", default=DET2_DEFAULT_DIR)
    train_ap.add_argument('-tensorboard',"--tensorboard",action='store_true', help="Start Tensorboard", default=False)
    
    test = subs.add_parser('test',help='test model')
    test.add_argument('-w',"--weights", type=str, default=os.path.join(DET2_DEFAULT_DIR, DET2_PT_EXPORT), help="The path to the model")
    test.add_argument('-i',"--input", type=str, default=DET2_INPUT_DIR, help="The path to the images")
    test.add_argument('-o',"--output", type=str, default=DET2_OUTPUT_DIR, help="The path to the outputs")
    test.add_argument("--class_map", type=str, default=DET2_CLASS_MAP, help="The path to the class map")
    test.add_argument("--confidence", type=float, default=0.5, help="The confidence threshold")
    test.add_argument("--iou", type=float, default=0.0, help="The IoU threshold for NMS")
    test.add_argument("--max_det", type=int, default=100, help="The max detections per image after NMS")
    
    convert_ap = subs.add_parser('convert',help='convert model')
    
    convert_ap.add_argument('-c',"--config-file",metavar="FILE", help="path to config file", default=os.path.join(DET2_DEFAULT_DIR, DET2_CONFIG_FILE))
    convert_ap.add_argument(
        "-o", "--output", help="The output directory for the converted model", type=str, default=DET2_DEFAULT_DIR)
    convert_ap.add_argument(
        "-w", "--weights", help="The Detectron 2 model weights (.pth)", type=str, default=os.path.join(DET2_DEFAULT_DIR, DET2_PTH_EXPORT),
    )
    convert_ap.add_argument(
        "-s", "--sample_image", help="Sample image for anchors generation/predictions", type=str, default=os.path.join(DET2_DEFAULT_DIR, 'sample_image.png'),
    )
    convert_ap.add_argument("-b","--batch-size", type=int, help="Batch size for the model", default=1)
    convert_ap.add_argument('-fp16',"--fp16",action='store_true', help="Use fp16", default=False)
    convert_ap.add_argument('-pt',"--pt",action='store_true', help="Convert to pt")
    convert_ap.add_argument('-onnx',"--onnx",action='store_true', help="Convert to onnx")
    convert_ap.add_argument('-trt', "--trt",action='store_true', help="Convert to TensorRT")
    
    args = ap.parse_args()
    args = vars(args)
    
    if args['action'] == 'train':
        logger.info("Training model")
        training_run(args)
    elif args['action'] == 'test':
        inference_run(args)
    elif args["action"] == 'convert':
        args['onnx_file_path'] = os.path.join(args.get('output'), DET2_ONNX_EXPORT)
        args['trt_file_path'] = os.path.join(args.get('output'), DET2_TRT_EXPORT)
        args['pt_file_path'] = os.path.join(args.get('output'), DET2_PT_EXPORT)
        convert(args)
