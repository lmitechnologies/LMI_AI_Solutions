#%%
# locate key feature
# crop center around same feature
# select golden reference
# align images to golden reference

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import glob
import shutil
import csv 
import tensorflow as tf
from yaml import load

import sys
sys.path.append('/app/LMI_AI_Solutions/lmi_utils')
sys.path.append('/app/LMI_AI_Solutions/anomaly_detectors')
sys.path.append('/app/LMI_AI_Solutions/object_detectors')
sys.path.append('/app/LMI_AI_Solutions/object_detectors/tf_objdet/models/research')
sys.path.append('/app/LMI_AI_Solutions/classifiers')
sys.path.append('/app/LMI_AI_Solutions/ocr_models')

from label_utils.csv_utils import load_csv
from padim.padim import plot_fig
from padim.padim import PaDiM
from padim.padim import DataLoader



# Relative data path
TF_VERSION="TF-2.11.0"
ANOMDET_DATA_PATH_TRAIN=os.path.join('/app/data/','train')
ANOMDET_DATA_PATH_TEST=os.path.join('/app/data/','test')
ANOMDET_MODEL_PATH=os.path.join('/app/trained-inference-models/',TF_VERSION)

def train_padim(basepath):
    cprime=200
    padim=PaDiM(GPU_memory=1024*8)
    dataloader=DataLoader(path_base=basepath,img_shape=(224,224),batch_size=1)
    #resnet layers
    layerconfig1={'layer1':'pool1_pool','layer2':'conv2_block1_preact_relu','layer3':'conv3_block1_preact_relu'}
    layerconfig2={'layer1':'conv2_block1_preact_relu','layer2':'conv3_block1_preact_relu','layer3':'conv4_block1_preact_relu'}
    layerconfig3={'layer1':'conv3_block1_preact_relu','layer2':'conv4_block1_preact_relu','layer3':'conv5_block1_preact_relu'}
    #efficientnet layers
    layerconfig4={'layer1':'stem_activation','layer2':'block2a_activation','layer3':'block3a_activation'}
    layerconfig5={'layer1':'block2a_activation','layer2':'block3a_activation','layer3':'block4a_activation'}
    layerconfig6={'layer1':'block3a_activation','layer2':'block4a_activation','layer3':'block5a_activation'}
    layerconfig7={'layer1':'block4a_activation','layer2':'block5a_activation','layer3':'block6a_activation'}

    padim.padim_train(dataloader,c=cprime,net_type='res',layer_names=layerconfig1,is_plot=True)      
    if not os.path.isdir(os.path.join(basepath,'saved_model')):
        os.makedirs(os.path.join(basepath,'saved_model'))
    padim.net.save(os.path.join(basepath,'saved_model','saved_model'))
    padim.export_tensors(fname=os.path.join(basepath,'saved_model','padim.tfrecords'))

def test_padim(testdatapath,modelpath,err_thresh):
    padim=PaDiM(GPU_memory=1024)
    padim.import_tfrecords(os.path.join(modelpath,'padim.tfrecords'))
    padim.net=tf.keras.models.load_model(os.path.join(modelpath,'saved_model'))
    image_tensor,dist_tensor,fname_tensor=padim.predict(os.path.join(testdatapath))
    err_dist_array=dist_tensor.numpy()
    image_array=image_tensor.numpy()
    fname_array=fname_tensor.numpy()

    prediction_results=zip(image_array,err_dist_array,fname_array)
    prediction_results_path=os.path.join(testdatapath,'prediction_results')
    if not os.path.isdir(prediction_results_path):
        os.makedirs(prediction_results_path)
    plot_fig(prediction_results,padim.training_mean_dist,padim.training_std_dist,prediction_results_path,err_thresh=err_thresh)

modelpath=ANOMDET_MODEL_PATH
#Train
if not os.path.exists(modelpath):
    os.makedirs(modelpath)
datapath=ANOMDET_DATA_PATH_TRAIN
train_padim(datapath)
shutil.move(os.path.join(datapath,'saved_model'),modelpath)
#Validate Training
saved_model_path=os.path.join(modelpath,'saved_model')
print(f'[INFO] Loading model from: {saved_model_path}')
print(f'[INFO] loading data from: {datapath}')
test_padim(datapath,saved_model_path,30)
shutil.move(os.path.join(datapath,'prediction_results'),os.path.join(datapath,'prediction_results_training'))
shutil.move(os.path.join(datapath,'prediction_results_training'),os.path.join(os.path.split(ANOMDET_MODEL_PATH)[0],TF_VERSION))
#Validate Testing
datapath=ANOMDET_DATA_PATH_TEST
saved_model_path=os.path.join(modelpath,'saved_model')
print(f'[INFO] Loading model from: {saved_model_path}')
print(f'[INFO] loading data from: {datapath}')
test_padim(datapath,saved_model_path,30)
shutil.move(os.path.join(datapath,'prediction_results'),os.path.join(datapath,'prediction_results_testing'))
shutil.move(os.path.join(datapath,'prediction_results_testing'),os.path.join(os.path.split(ANOMDET_MODEL_PATH)[0],TF_VERSION))

