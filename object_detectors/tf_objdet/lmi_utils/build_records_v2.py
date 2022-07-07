#%% import packages
from tfannotation import TFAnnotation
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

# TODO: 
# expand to support keypoints
# remove Rosenbrock dependency

#%%
def main(config):

    if not os.path.exists(os.path.split(config.CLASSES_FILE)[0]):
        os.makedirs(os.path.split(config.CLASSES_FILE)[0])

    f=open(config.CLASSES_FILE,'w')

    # Catch errors for unspecified config parameters
    # try:
    #     MASK_OPTION=config.MASK_OPTION
    # except:
    #     print('[INFO] Faster R-CNN by default since MASK_OPTION is not defined in ',args['config_path'])
    #     MASK_OPTION=False   
    try:
        RESIZE_OPTION=config.RESIZE_OPTION
    except:
        print('[INFO] No resizing because RESIZE_OPTION is not defined in ',args['config_path'])
        RESIZE_OPTION=False
    try:
        keypoints=config.KEYPOINTS
        keypoint_num=0
        KEYPOINT_OPTION=True
    except:
        print('No keypoint definition.  Skipping all keypoint features.')
        KEYPOINT_OPTION=False

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
        # if MASK_OPTION:
        if row[2]=='polygon':
            if row[3]=='x values':
                imagePath=row[0]
                label=row[1]
                xvec=np.array(row[4:],dtype=np.float)
                startX=xvec.min()
                endX=xvec.max()
                continue
            if row[3]=='y values':
                yvec=np.array(row[4:],dtype=np.float)
                startY=yvec.min()
                endY=yvec.max()
                is_mask=True
        elif row[2]=='rect':
            if row[3]=='upper left':
                (imagePath,label,_,_,startX,startY)=row
                (startX,startY)=(float(startX),float(startY))
                continue
            if row[3]=='lower right':
                (endX,endY)=(row[4],row[5]) 
                (endX,endY)=(float(endX),float(endY))
                is_bbox=True
        elif row[2]=='point':
            if row[3]=='cx':
                kp_label
                cx=float(row[4])
                continue
            if row[3]=='cy':
                cy=float(row[4])
                is_keypoint=True
        else:
            raise Exception(f'Unregonized feature: {row[2]}.  This conversion only supports: polygon,rect,point')

        # optional:ignore label if not interested
        if label not in config.CLASSES:
            print('[INFO] Skipping class: ',label)
            continue

        #build path to input image, then grab any other bounding boxes + labels associated with the image path, labels, bounding box lists, respectively
        p=os.path.sep.join([config.DATA_PATH,imagePath])
        b=D.get(p,[])

        #build tuple consisting of the label and bounding box, then update the list and store it in the dictionary
        if is_mask:
            b.append((label,('mask',startX,startY,endX,endY,xvec,yvec)))
            project_mask_option=True
        elif is_bbox:
            b.append((label,('bbox',startX,startY,endX,endY)))
        elif is_keypoint:
            b.append((label,('keypoint',cx,cy)))
            project_keypoint_option=True
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
                    MAX_W=config.MAX_W
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
                        #skip resize if w<MAX_W
                        RESIZE_OPTION=False
                        encoded=tf.io.gfile.GFile(k,'rb').read()
                        encoded=bytes(encoded)              
                except:
                    print('[INFO] No resizing because MAX_W is not defined in ',args['config_path'])
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
         
            #load the image from disk again, this time as a PIL object
            # pilImage=Image.open(k)
            # (w,h)=pilImage.size[:2]

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
            
            # loop over image bounding boxes + labels
            # (type,startX,startY,endX,endY,xvec,yvec)
            for (label,annot) in D[k]:
                tfAnnot.textLabels.append(label.encode('utf8'))
                tfAnnot.classes.append(config.CLASSES[label])
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
                    kn=keypoint_num


                
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
