#%% import packages
from tf_objdet.lmi_utils.tfannotation import TFAnnotation
from sklearn.model_selection import train_test_split
from PIL import Image
import tensorflow as tf 
import os
import sys
import argparse
from runpy import run_path
import importlib.util
import ast
import numpy as np
import cv2
import io
from image_utils.img_resize import resize

# get item in a recursive dictionary 
def _finditem(obj, key):
    if key in obj: return obj[key]
    for k, v in obj.items():
        if isinstance(v,dict):
            item = _finditem(v, key)
            if item is not None:
                return item

def main(config):

    if not os.path.exists(os.path.split(config.CLASSES_FILE)[0]):
        os.makedirs(os.path.split(config.CLASSES_FILE)[0])

    f=open(config.CLASSES_FILE,'w')

    # Load training options
    # masks
    try:
        MASK_OPTION=config.MASK_OPTION
    except:
        print(f'[INFO] MASK_OPTION is not defined in config file.  Finding bounding boxes.')
        MASK_OPTION=False
    # keypoints
    keypoints=None
    keypoint_num=None
    keypoint_ids=None
    keypoint_names=None
    try:
        KEYPOINT_OPTION=config.KEYPOINT_OPTION
    except:
        print(f'[INFO] KEYPOINT_OPTION is not defined in config file.  Ignoring keypoints.')
        KEYPOINT_OPTION=False 
    if KEYPOINT_OPTION:
        try:
            keypoints=config.KEYPOINTS
            keypoint_num=0
            keypoint_ids=[]
            keypoint_names=[]
        except:
            print(f'[INFO] KEYPOINTS not defined in config file.  Ignoring keypoints.')
            KEYPOINT_OPTION=False
    
    # resize
    try:
        RESIZE_OPTION=config.RESIZE_OPTION
    except:
        print(f'[INFO] RESIZE_OPTION is not defined in config file.  No resizing applied.')
    if RESIZE_OPTION:
        try:
            MAX_W=config.MAX_W
        except:
            print(f'[INFO] MAX_W is not defined in config file.  No resizing applied.')
            RESIZE_OPTION=False


    # loop over classes and place labels in a JSON-like file
    # required keys:id, name
    # start counting at 1, class 0 reserved for background
    for (k,v) in config.CLASSES.items():
        # construct the class information and write to file
        item = ("item {\n"
            "\tid: " + str(v) + "\n"
            "\tname: '" + k + "'\n")
        f.write(item)
        if KEYPOINT_OPTION:
            for (kk0,kv0) in keypoints.items():
                if kk0==k:
                    for (kk1,kv1) in kv0.items():
                        keypoint_num+=1
                        keypoint_names.append(kk1)
                        keypoint_ids.append(kv1)
                        kpi=("\tkeypoints: {\n"
                        "\t\tid: " + str(kv1) +"\n"
                        "\t\tlabel: '" + str(kk1) +"'\n"
                        "\t}\n")
                        f.write(kpi)            
        f.write("}\n")
    f.close()

    # initialize a data dictionary used to map each image filename to all bounding boxes associated with the image, then load the contents of the annotations file
    D={}
    # parse .csv file
    # create dictionary, keys=images, value=payload:label, bounding box
    # randomize the training and testing split
    rows=open(config.ANNOT_PATH).read().strip().split('\n')
    for row in rows[0:]:
        is_mask, is_bbox, is_keypoint=False, False, False
        print('[INFO] row:',row)
        row=row.split(';')
        if row[3]=='rect':
            if row[4]=='upper left':
                (imagePath,label,_,_,_,startX,startY)=row
                (startX,startY)=(float(startX),float(startY))
                continue
            if row[4]=='lower right':
                (endX,endY)=(row[5],row[6]) 
                (endX,endY)=(float(endX),float(endY))
                is_bbox=True
        elif row[3]=='polygon':
            if MASK_OPTION:  
                if row[4]=='x values':
                    imagePath=row[0]
                    label=row[1]
                    xvec=np.array(row[5:],dtype=np.float)
                    startX=xvec.min()
                    endX=xvec.max()
                    continue
                if row[4]=='y values':
                    yvec=np.array(row[5:],dtype=np.float)
                    startY=yvec.min()
                    endY=yvec.max()
                    is_mask=True
            else:
                continue
        elif row[3]=='point':
            if KEYPOINT_OPTION:
                if row[4]=='cx':
                    label=row[1]
                    cx=float(row[5])
                    continue
                if row[4]=='cy':
                    cy=float(row[5])
                    is_keypoint=True
            else:
                print(f'[INFO] KEYPOINT_OPTION set to false in config file.  Skipping Keypoint.')
                continue
        else:
            raise Exception(f'Unregonized feature: {row[3]}.  This conversion only supports: polygon,rect,point')

        # optional:ignore label if not interested
        if (label not in config.CLASSES) and (label not in keypoint_names):
            print('[INFO] Skipping class: ',label)
            continue

        #build path to input image, then grab any other bounding boxes + labels associated with the image path, labels, bounding box lists, respectively
        p=os.path.sep.join([config.DATA_PATH,imagePath])
        b=D.get(p,[])

        #build tuple consisting of the label and bounding box, then update the list and store it in the dictionary
        if is_mask:
            b.append((label,('mask',startX,startY,endX,endY,xvec,yvec)))
        elif is_bbox:
            b.append((label,('bbox',startX,startY,endX,endY)))
        elif is_keypoint:
            b.append((label,('keypoint',cx,cy)))
        D[p]=b
    
    # create training and testing splits from data dictionary
    (trainKeys,testKeys)=train_test_split(list(D.keys()),test_size=config.TEST_SIZE,random_state=42)
    # initialize datasplit files:
    datasets=[
        ('train',trainKeys,config.TRAIN_RECORD),
        ('test',testKeys,config.TEST_RECORD)
        ]
    
    # build tensorflow record files
    # loop over training and testing splits
    for (dType,keys,outputPath) in datasets:
        # initialize the TensorFlow writer and total #of examples written to file
        print('[INFO] processing "{}"...'.format(dType))
        writer=tf.io.TFRecordWriter(outputPath)
        total=0
        # loop over all keys in the current set

        for k in keys:
            img=cv2.imread(k)
            (h0,w0)=img.shape[:2]
            # resize input image
            if RESIZE_OPTION:
                try:
                    if w0>MAX_W:
                        img_resize=resize(img,width=MAX_W)
                        (h,w)=img_resize.shape[:2]
                        print(f'[INFO] Resized input image, bounding boxes, and polygons to image size w={w}, h={h}.')
                        img_rgb=cv2.cvtColor(img_resize, cv2.COLOR_BGR2RGB)
                        img_pil=Image.fromarray(img_rgb)
                        output = io.BytesIO()
                        img_pil.save(output,format='PNG')
                        encoded=output.getvalue()
                    else:
                        print(f'[INFO] Image w={w0} is less than MAX_W={MAX_W}.  Skipping resize.')
                        encoded=tf.io.gfile.GFile(k,'rb').read()
                        encoded=bytes(encoded)              
                except:
                    print(f'[INFO] No resizing because MAX_W is not correctly defined in config file.')
                    # if MAX_W is undefined, then skip resize
                    RESIZE_OPTION=False
                    encoded=tf.io.gfile.GFile(k,'rb').read()
                    encoded=bytes(encoded)
                    continue
            else:
                encoded=tf.io.gfile.GFile(k,'rb').read()
                encoded=bytes(encoded)

            if not RESIZE_OPTION:
                h=h0
                w=w0

            #parse the filename and encoding from the input path
            filename=k.split(os.path.sep)[-1]
            encoding=filename[filename.rfind('.')+1:]

            #initialize the annotation object used to store bounding box and label info
            tfAnnot=TFAnnotation()

            tfAnnot.image=encoded
            tfAnnot.encoding=encoding
            tfAnnot.filename=filename
            tfAnnot.width=w
            tfAnnot.height=h
            if KEYPOINT_OPTION:
                tfAnnot.is_keypoint=True
                tfAnnot.num_keypoints=[keypoint_num]
                tfAnnot.keypoints_visibility=[0]*keypoint_num
                tfAnnot.keypoints_x=[0]*keypoint_num
                tfAnnot.keypoints_y=[0]*keypoint_num
                tfAnnot.keypoints_name=[name.encode('utf8') for name in keypoint_names]

            
            # loop over image bounding boxes + labels
            # (type,startX,startY,endX,endY,xvec,yvec)
            for (label,annot) in D[k]:
                try:
                    tfAnnot.classes.append(config.CLASSES[label])
                    tfAnnot.textLabels.append(label.encode('utf8'))
                except:
                    assert (label in keypoint_names)
                    
                if annot[0]=='bbox' or annot[0]=='mask':
                # TensorFlow requires normalized bounding boxes
                # Normalized bounding boxes don't need resizing
                    xMin=annot[1]/w0
                    xMax=annot[3]/w0
                    yMin=annot[2]/h0
                    yMax=annot[4]/h0

                    # update bounding boxes and label lists
                    tfAnnot.xMins.append(xMin)
                    tfAnnot.xMaxs.append(xMax)
                    tfAnnot.yMins.append(yMin)
                    tfAnnot.yMaxs.append(yMax)
                    # tfAnnot.difficult.append(0)
                if annot[0]=='mask':
                    xvec=annot[5]
                    yvec=annot[6]
                    tfAnnot.is_mask=True
                    canvas=np.zeros((h0,w0),dtype=np.uint8)
                    pts=np.stack((xvec,yvec),axis=1)
                    pts=np.expand_dims(pts,axis=0).astype(np.int32)
                    mask_img=cv2.fillPoly(canvas,pts,1).astype(np.uint8)
                    if RESIZE_OPTION and w0>MAX_W:
                        img_resize=resize(mask_img,width=MAX_W,inter=cv2.INTER_NEAREST)
                        mask_img=img_resize
                    img=Image.fromarray(mask_img)
                    output = io.BytesIO()
                    img.save(output,format='PNG')
                    tfAnnot.masks.append(output.getvalue())  
                if annot[0]=='keypoint':
                    kx=annot[1]/w0
                    ky=annot[2]/h0
                    index=_finditem(config.KEYPOINTS,label)
                    tfAnnot.keypoints_x[index]=kx
                    tfAnnot.keypoints_y[index]=ky
                    tfAnnot.keypoints_visibility[index]=1
                total+=1

            # encode the data point attributes using the TensorFlow helper functions
            features=tf.train.Features(feature=tfAnnot.build())
            example=tf.train.Example(features=features)
            # add the example to the writer
            writer.write(example.SerializeToString())
        # close writer adn show diagnostics
        writer.close()
        print('[INFO] {} examples saved for "{}"'.format(total,dType))

# check for main
if __name__=='__main__':
    # add command line arguments for config file
    ap=argparse.ArgumentParser()
    ap.add_argument('-c','--config_path',required=True,help='path to config.py')
    args=vars(ap.parse_args())
    config_path=args['config_path']

    config_name=os.path.splitext(os.path.split(config_path)[1])[0]
    spec = importlib.util.spec_from_file_location(config_name,config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    main(config)
