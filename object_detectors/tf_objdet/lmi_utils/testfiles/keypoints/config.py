import os

#%% Define paths for data
DATA_PATH='./testfiles/keypoints/data'
ANNOT_PATH=os.path.sep.join([DATA_PATH,'labels.csv'])

#%% Define paths for records
TRAIN_RECORD='./testfiles/keypoints/records/training.record'
TEST_RECORD='./testfiles/keypoints/records/testing.record'
CLASSES_FILE='./testfiles/keypoints/records/classes.pbtxt'

#%% Initialize training/test split
TEST_SIZE=0.2
RESIZE_OPTION=False

#%% initialize the class labels dictionary
CLASSES={'hog':1,'leg':2}
KEYPOINTS={'leg':{'hoof':0,'aitch-bone':1}}
