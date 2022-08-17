from numpy.core.fromnumeric import shape
import torch
TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)

# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, cv2

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
    

def register_datasets(cfg):
    #register test data
    for dataset_name, dataset_path in zip(cfg.DATASETS.TEST, cfg.DATASETS.TEST_DIR):
        path_json_labels = os.path.join(dataset_path,"labels.json")
        if not os.path.isfile(path_json_labels):
            raise Exception("cannot find the annotation file labels.json in {}".format(dataset_path))
        register_coco_instances(dataset_name, {}, path_json_labels, dataset_path)
    


def test(cfg):
    #path to the model we just trained
    cfg.MODEL.WEIGHTS = os.path.join(cfg.TRAINED_MODEL_DIR, "model_final.pth")  
    predictor = DefaultPredictor(cfg)
    #inference
    import time
    from detectron2.utils.visualizer import ColorMode, GenericMask
    for dataset_name in cfg.DATASETS.TEST:
        dataset_dicts = DatasetCatalog.get(dataset_name)
        meta_data = MetadataCatalog.get(dataset_name)
        out_path = os.path.join(cfg.OUTPUT_DIR, dataset_name)
        os.makedirs(out_path,exist_ok=True)
        proc_times = []
        pred_list = []
        for d in dataset_dicts:    
            im = cv2.imread(d["file_name"])
            im_name = os.path.basename(d["file_name"])

            st = time.time()
            outputs = predictor(im) 
            et = time.time()
            proc_times.append(et-st)
            print('[INFO]proc time: {:.4f}'.format(et-st))

            v = Visualizer(im[:, :, ::-1],
                metadata=meta_data, 
                scale=2, 
                instance_mode=ColorMode.IMAGE_BW  
            )
            out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            out.save(os.path.join(out_path, im_name))
            
            #prepare for csv outputs
            instances = outputs["instances"]
            h,w = im.shape[:2]
            #loop through each instance (mask)
            for mask,bbox,c,score in zip(instances.pred_masks, instances.pred_boxes, instances.pred_classes, instances.scores):
                mask = mask.cpu().numpy()
                bbox = bbox.cpu().numpy()
                score = score.item()
                c = c.item()
                GM = GenericMask(mask,h,w)
                #merge multiple polygons
                X,Y = np.array(()),np.array(())
                for poly in GM.polygons:
                    poly2d = poly.reshape((-1,2))
                    x,y = poly2d[:,0], poly2d[:,1]
                    X = np.concatenate((X,x))
                    Y = np.concatenate((Y,y))
                X = X.astype(np.int)
                Y = Y.astype(np.int)
                bbox = bbox.astype(np.int)
                temp_dt = {'name':im_name,'class':meta_data.thing_classes[c],'conf':score,'bbox':bbox.tolist(), 'x':X.tolist(), 'y':Y.tolist()}
                pred_list.append(temp_dt)

        #generate output csv file
        import csv
        with open(os.path.join(out_path,'preds.csv'), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=';')
            for dt in pred_list:
                name = dt['name']
                class_name = dt['class']
                bbox = dt['bbox']
                conf = dt['conf']
                x,y = dt['x'],dt['y']
                writer.writerow([name,class_name,conf,'rect','upper left']+bbox[:2])
                writer.writerow([name,class_name,conf,'rect','lower right']+bbox[2:])
                writer.writerow([name,class_name,conf,'polygon','x values']+x)
                writer.writerow([name,class_name,conf,'polygon','y values']+y)
        proc_times = np.array(proc_times[1:]) 
        print('[INFO]average proc time: {}'.format(proc_times.mean()))
        print('[INFO]min proc time: {}'.format(proc_times.min()))
        print('[INFO]max proc time: {}'.format(proc_times.max()))



if __name__=='__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--input', required=True, type=str, help='the input yaml file')
    ap.add_argument('--base_yaml', default="mask_rcnn_R_50_C4_1x.yaml", help='the base yaml file provided in https://github.com/facebookresearch/detectron2/tree/main/configs/COCO-InstanceSegmentation')
    args = vars(ap.parse_args())

    cfg = get_cfg()
    #create new customized keys in config file
    cfg.DATASETS.TEST_DIR = ()
    cfg.TRAINED_MODEL_DIR = ""
    #load yaml files
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/"+args['base_yaml']))
    cfg.merge_from_file(args['input'])
    #register train and test datasets
    register_datasets(cfg)
    #testing
    test(cfg)