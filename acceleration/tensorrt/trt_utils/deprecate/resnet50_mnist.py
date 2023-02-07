#%% dependencies
import argparse
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import time
from tensorflow.python.compiler.tensorrt import trt_convert as trt
import os
import copy

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


#normalization function maps png to normalized float
def normalize_img(image, label):
  """Resizes and Normalizes images: `uint8` -> `float32`."""
  image=tf.image.resize(image,[32,32])
  return tf.cast(image, tf.float32) / 255., label


# %% design resnet model
def design_resnet50(input_shape=(32,32,1)):
    '''
    DESCRIPTION: Imports ResNet50 and adds new fully connected layer for classification

    ARGS: 
        -input_shape (h,w,ch) 
    '''
    model=tf.keras.models.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=input_shape))
    # model.add(tf.keras.layers.experimental.preprocessing.Rescaling(1/127.5,offset=-1))
    base_model=tf.keras.applications.ResNet50(include_top=False, weights=None, input_shape=input_shape)
    model.add(base_model)
    model.add(tf.keras.layers.GlobalAveragePooling2D())
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(10,activation='softmax'))
    base_model.trainable=True
    model.summary()
    return model

def run_benchmark(loaded_model,ds_bench):
    '''
    DESCRIPTION: Benchmarks processing time for batch size = 1

    ARGUMENTS: 
        -loaded_model: TensorFlow model object
        -image_path_list: list of paths to .png files used for inference
    '''
    max_iter=100
    ptime=[]
    iter=1
    classify_fn=loaded_model.signatures['serving_default']
    for batch_image,batch_label in tfds.as_numpy(ds_bench):
        zipped=zip(batch_image,batch_label)
        for image,label in zipped:
            print(f'[INFO] Iteration: {iter}')
            t0=time.time()
            results=classify_fn(tf.convert_to_tensor(np.expand_dims(image,axis=0)))
            decision=np.argmax(results['dense'])
            print(f'[INFO] Predicted: {decision}, Actual: {label}')
            t1=time.time()
            tdelta=t1-t0
            print(f'[INFO] Proc Time: {tdelta}')
            image_x=(image*255.0).astype(np.uint8)
            if iter==max_iter:
                break
            else:
                ptime.append(tdelta)
                iter=iter+1
        if iter==max_iter:
            break
    ptime=np.array(ptime)
    ptime=np.sort(ptime)

    print(f'*****************************************')
    print(f'[INFO] Average runtime: {np.mean(ptime[1:-1])}')
    print(f'[INFO] Min runtime: {np.min(ptime[1:-1])}')
    print(f'[INFO] Max runtime: {np.max(ptime[1:-1])}')
    print(f'*****************************************') 

def convert_tensorRT(saved_model_dir,trt_saved_model_dir,precision_mode='FP16'):
    '''
    DESCRIPTION: Converts a saved_model to a tensorRT saved model.  Engine is built each time at runtime.

    ARGUMENTS:
        -saved_model_dir: path to input tensorflow saved_model
        -trt_saved_model_dir: path to output tensorflow saved_model  
        -precision_mode: fixed point precision used by converter 
    '''
    # Define key properties
    # https://docs.nvidia.com/deeplearning/frameworks/tf-trt-user-guide/index.html
    params = copy.deepcopy(trt.DEFAULT_TRT_CONVERSION_PARAMS)
    precision_mode='FP16'
    allow_build_at_runtime=True # Default:True
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
        precision_mode=get_trt_precision()
    )
    #%% Convert the SavedModel to tensorRT
    converter = trt.TrtGraphConverterV2(input_saved_model_dir=saved_model_dir,conversion_params=params)
    converter.convert()
    # Save the converted model
    converter.save(trt_saved_model_dir)

def main(args):

    set_gpu(args['gpu_mem_limit'])

    TRAINED_MODEL_PATH='./trained-inference-models'
    if os.path.isdir(TRAINED_MODEL_PATH):
        pass
    else:
        os.makedir(TRAINED_MODEL_PATH)

    #%% load mnist dataset from tensorflow datasets
    (ds_train, ds_test), ds_info = tfds.load(
        'mnist',
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )

    # preprocess training/test data
    # training
    ds_train = ds_train.map(
        normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
    ds_train = ds_train.batch(64)
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)
    # test
    ds_test = ds_test.map(
        normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_test = ds_test.batch(64)
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)
    #design model
    if args['design_baseline']:
        model=design_resnet50()
        #configure training
        model.compile(
            optimizer=tf.keras.optimizers.Adam(0.001),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
        )
        #configure callbacks 
        tensorboard_callback=tf.keras.callbacks.TensorBoard()
        checkpoint_filepath='./tmp/checkpoint'
        monitor_request='val_accuracy'    
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            monitor=monitor_request,
            mode='max',
            save_best_only=True)
        training_callbacks=[model_checkpoint_callback,tensorboard_callback]
        #train model
        model.fit(
            ds_train,
            epochs=2,
            validation_data=ds_test,
            callbacks=training_callbacks,
            verbose=1)
        #save trained model
        model.save(os.path.join(TRAINED_MODEL_PATH,args['baseline_saved_model_dir']))
    
    baseline_saved_model_dir=os.path.join(TRAINED_MODEL_PATH,args['baseline_saved_model_dir'])
    trt_model_dir=os.path.join(TRAINED_MODEL_PATH,args['trt_saved_model_dir'])
    if args['generate_trt']:    
        convert_tensorRT(baseline_saved_model_dir,trt_model_dir)
        tf.keras.backend.clear_session()
    if args['benchmark_baseline']:
        loaded_model=tf.keras.models.load_model(baseline_saved_model_dir)
        run_benchmark(loaded_model,ds_test)
        tf.keras.backend.clear_session()
    if args['benchmark_trt']:
        loaded_model=tf.keras.models.load_model(trt_model_dir)
        run_benchmark(loaded_model,ds_test)
        tf.keras.backend.clear_session()


# %%
if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--gpu_mem_limit',default=2048,help='GPU memory limit (MB)')
    ap.add_argument('--baseline_saved_model_dir',default='resnet_50_saved_model')
    ap.add_argument('--trt_saved_model_dir',default='resnet_50_saved_model_rt')
    ap.add_argument('--design_baseline',dest='design_baseline',action='store_true')
    ap.add_argument('--generate_trt',dest='generate_trt',action='store_true')
    ap.add_argument('--benchmark_baseline',dest='benchmark_baseline',action='store_true')
    ap.add_argument('--benchmark_trt',dest='benchmark_trt',action='store_true')
    ap.set_defaults(design_baseline=False,generate_trt=False,benchmark_baseline=False,benchmark_trt=False)

    args=vars(ap.parse_args())
    
    main(args)
