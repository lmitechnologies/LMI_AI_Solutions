import os
import sys
import argparse
import glob
import shutil
import subprocess
import re

sys.path.append('/app/LMI_AI_Solutions/lmi_utils')
sys.path.append('/app/LMI_AI_Solutions/anomaly_detectors')
sys.path.append('/app/LMI_AI_Solutions/object_detectors')
sys.path.append('/app/LMI_AI_Solutions/object_detectors/tf_objdet/models/research')
sys.path.append('/app/LMI_AI_Solutions/classifiers')
sys.path.append('/app/LMI_AI_Solutions/ocr_models')
sys.path.append('/app/LMI_AI_Solutions/acceleration/tensorrt')
sys.path.append('/app/LMI_AI_Solutions/acceleration/tensorrt/tftrt/benchmarking-python')

from trt_utils.trt_objdet.savedmodel_withbuild import main as convert_trt
from padim.padim import main as convert_trt_padim, PaDiM

from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants
from tensorflow import float32, uint8

INPUT_DTYPES = {'DT_UINT8': uint8, 'DT_FLOAT': float32, 'None': None, 'none': None}

def main(params):
    args={
        'gpu_mem_limit':3072,
        'baseline_saved_model_dir':'/app/model',
        'trt_saved_model_dir':'/app/output',
        'data_dir':'/app/images',
        'cal_data_dir': '/app/images-calibration'
    }
    args.update(params)

    if not os.path.isdir(args['cal_data_dir']):
        os.mkdir(args['cal_data_dir'])
    shutil.copyfile(glob.glob(os.path.join(args['data_dir'],'*.png'))[0], f"{args['cal_data_dir']}/calibration_image.png")

    show_model_cmd = f"saved_model_cli show --dir {args['baseline_saved_model_dir']}/saved_model --tag_set {tag_constants.SERVING} --signature_def {signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY}"
    output = subprocess.check_output(show_model_cmd, shell=True)
    res = re.search("\s+dtype:\s*([\w_]+)", output.decode())
    if not res:
        print(f"No input_dtype was found - {output}")
    args['input_dtype'] = INPUT_DTYPES[res.groups()[0]] if res else 'None'
    
    if PaDiM.get_tfrecords(args['baseline_saved_model_dir']):
        convert_trt_padim(args)
    else:
        convert_trt(args)

if __name__ == '__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--generate_trt',dest='generate_trt',action='store_true')
    ap.add_argument('--benchmark_baseline',dest='benchmark_baseline',action='store_true')
    ap.add_argument('--benchmark_trt',dest='benchmark_trt',action='store_true')
    ap.set_defaults(generate_trt=False,benchmark_baseline=False,benchmark_trt=True)
    args=vars(ap.parse_args())

    main(args)