#%%  Launch docker 
# docker run -it --rm    --gpus="all"    --shm-size=2g --ulimit memlock=-1 --ulimit stack=67108864    --workdir /workspace/    -v "$(pwd):/workspace/"   nvcr.io/nvidia/tensorflow:21.10-tf2-py3

#%% Dependencies
import glob
import os
import copy
import time

import argparse
import numpy as np

# import before tf to avoid "cannot allocate memory in static TLS block"
import cv2

import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt

#%% gpu configuration
def set_gpu(gpu_mem_limit):
    '''
    DESCRIPTION: Sets GPU limit.

    ARGUMENTS:
        -gpu_mem_limit: memory limit in MBs
    
    '''
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=gpu_mem_limit)])
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)

def preprocess_image(image_path, dtype=None):
    '''
    DESCRIPTION: preprocess images

    ARGUMENTS:
        -image_path: image path to .png file

    RETURNS:
        -image tensor
    '''
    image=cv2.imread(image_path)
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image=np.expand_dims(image,axis=0)
    image_tensor=tf.convert_to_tensor(image, dtype=dtype)
    return image_tensor

def run_benchmark(loaded_model,image_path_list,input_dtype):
    '''
    DESCRIPTION: Benchmarks processing time for batch size = 1

    ARGUMENTS: 
        -loaded_model: TensorFlow model object
        -image_path_list: list of paths to .png files used for inference
    '''
    ptime=[]
    detect_fn=loaded_model.signatures['serving_default']
    for i,image_path in enumerate(image_path_list):
        image_tensor=preprocess_image(image_path, input_dtype)
        t0=time.time()
        detections = detect_fn(image_tensor)
        t1=time.time()
        tdelta=t1-t0
        ptime.append(tdelta)
        print('[INFO] Processing Time: ',tdelta, 's.')

    print(f'*****************************************')
    print(f'[INFO] Average runtime: {np.mean(ptime)}')
    print(f'[INFO] Median runtime: {np.median(ptime)}')
    print(f'[INFO] Min runtime: {np.min(ptime)}')
    print(f'[INFO] Max runtime: {np.max(ptime)}')
    print(f'*****************************************')


def convert_tensorRT(saved_model_dir,trt_saved_model_dir,cal_data_dir,input_dtype,precision_mode='FP16'):
    '''
    DESCRIPTION: Converts a saved_model to a tensorRT saved model.  Also builds the engine that gets loaded at runtime if calibration data directory is specified.

    ARGUMENTS:
        -saved_model_dir: path to input tensorflow saved_model
        -trt_saved_model_dir: path to output tensorflow saved_model  
        -cal_data_dir: path to calibration data directory used to prebuild executable
        -precision_mode: fixed point precision used by converter
    '''
    # Define key properties
    # https://docs.nvidia.com/deeplearning/frameworks/tf-trt-user-guide/index.html
    params = copy.deepcopy(trt.DEFAULT_TRT_CONVERSION_PARAMS)
    precision_mode='FP16'
    allow_build_at_runtime=False # Default:True
    max_workspace_size_bytes=1073741824 # Default: 1e9. The maximum GPU temporary memory which the TensorRT engine can use at execution time
    minimum_segment_size=3 # Default: 3. This is the minimum number of nodes required for a subgraph to be replaced by TRTEngineOp.
    # Set precision
    def get_trt_precision():
        if precision_mode == "FP32":
            return trt.TrtPrecisionMode.FP32
        elif precision_mode == "FP16":
            return trt.TrtPrecisionMode.FP16
        elif precision_mode == "INT8":
            return trt.TrtPrecisionMode.INT8
        else:
            raise RuntimeError("Unknown precision received: `{}`. Expected: "
                                "FP32, FP16 or INT8")

    # Set key properties
    params = params._replace(
        allow_build_at_runtime=allow_build_at_runtime,
        max_workspace_size_bytes=max_workspace_size_bytes,
        minimum_segment_size=minimum_segment_size,
        precision_mode=get_trt_precision(),
        use_calibration=False
    )

    print(f'[INFO] TRT Params: {params}')

    converter = trt.TrtGraphConverterV2(input_saved_model_dir=saved_model_dir,conversion_params=params)
    # Convert for FP16,FP32
    converter.convert()
    # TODO: add support for INT8: converter.convert(calibration_input_fn=calibration_input_fn)
    
    # Build
    if not allow_build_at_runtime:
        try:
            image_path_list=glob.glob(os.path.join(cal_data_dir,'*.png'))
            cal_image_list=[]
            for image_path in image_path_list:
                image_tensor=preprocess_image(image_path, input_dtype)
                cal_image_list.append(image_tensor)
            def calibration_input_fn():
                for x in cal_image_list:
                    print(f'Calibration image shape: {x.shape}')
                    yield [x]
            converter.build(input_fn=calibration_input_fn)
        except Exception as e:
            print('Calibration data directory is not specified properly.', e)

    converter.summary()
    # Save the converted model
    converter.save(trt_saved_model_dir)

def main(args):

    set_gpu(args['gpu_mem_limit'])

    input_dtype = INPUT_DTYPES[args['input_dtype']]

    image_path_list = []
    if 'data_dir' in args:
        path_to_test_images=args['data_dir']
        image_path_list=glob.glob(os.path.join(path_to_test_images,'*.png'))

    baseline_saved_model_dir=args['baseline_saved_model_dir']
    if os.path.isdir(os.path.join(baseline_saved_model_dir, "saved_model")):
        baseline_saved_model_dir = os.path.join(baseline_saved_model_dir, "saved_model")
    trt_model_dir=args['trt_saved_model_dir']
    if args['generate_trt']:    
        convert_tensorRT(baseline_saved_model_dir,trt_model_dir,args['cal_data_dir'],input_dtype)
        tf.keras.backend.clear_session()
    if args['benchmark_baseline']:
        loaded_model=tf.saved_model.load(baseline_saved_model_dir)
        run_benchmark(loaded_model,image_path_list,input_dtype)
    if args['benchmark_trt']:
        loaded_model=tf.saved_model.load(trt_model_dir)
        run_benchmark(loaded_model,image_path_list,input_dtype)

# %%
if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--gpu_mem_limit',default=3000,type=int,help='GPU memory limit (MB)')
    ap.add_argument('--baseline_saved_model_dir',default='/app/trained-inference-models/fasterrcnn_400x400/2021-12-08_400_rebuild_tf230/saved_model')
    ap.add_argument('--trt_saved_model_dir',default='/app/trained-inference-models/fasterrcnn_400x400/2021-12-08_400_rebuild_tf230/saved_model_rt')
    ap.add_argument('--data_dir',default='/app/data/testdata_400x400/training')
    ap.add_argument('--input_dtype',default=None)
    ap.add_argument('--cal_data_dir',default='/app/data/testdata_400x400/trt_calibration')
    ap.add_argument('--generate_trt',dest='generate_trt',action='store_true')
    ap.add_argument('--benchmark_baseline',dest='benchmark_baseline',action='store_true')
    ap.add_argument('--benchmark_trt',dest='benchmark_trt',action='store_true')
    ap.set_defaults(generate_trt=False,benchmark_baseline=False,benchmark_trt=True)

    args=vars(ap.parse_args())
    
    main(args)