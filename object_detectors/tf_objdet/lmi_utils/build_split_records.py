#%% import packages
from tf_objdet.lmi_utils.tfannotation import TFAnnotation
from sklearn.model_selection import train_test_split
from PIL import Image
import tensorflow as tf 
import os
import json
import csv
import argparse
import importlib.util
import numpy as np
import cv2
import io
from image_utils.img_resize import resize


#%%
def load_csv(path_imgs, path_csv, CLASSES, MASK_OPTION):
    # initialize a data dictionary used to map each image filename to all bounding boxes associated with the image, then load the contents of the annotations file
    D={}
    # parse .csv file
    # create dictionary, keys=images, value=payload:label, bounding box
    print(f'[INFO] loading the csv file: {path_csv}')
    with open(path_csv) as f:
        rows = csv.reader(f, delimiter=';')
        for row in rows:
            #print('[INFO] row:',row)
            if MASK_OPTION:
                if row[3]=='polygon':
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
                else:
                    raise Exception('csv file includes non-polygon regions.')
            else:
                if row[3]=='rect':
                    if row[4]=='upper left':
                        (imagePath,label,_,_,startX,startY)=row
                        (startX,startY)=(float(startX),float(startY))
                        continue
                    if row[4]=='lower right':
                        (endX,endY)=(row[5],row[6]) 
                        (endX,endY)=(float(endX),float(endY))
                elif row[3]=='polygon':
                    print('[INFO] Converting ROIs of type "polygon" to bounding boxes.')
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


            # optional:ignore label if not interested
            if label not in CLASSES:
                print('[INFO] Skipping class: ',label)
                continue

            #build path to input image, then grab any other bounding boxes + labels associated with the image path, labels, bounding box lists, respectively
            p=os.path.join(path_imgs,imagePath)
            b=D.get(p,[])

            #build tuple consisting of the label and bounding box, then update the list and store it in the dictionary
            if MASK_OPTION:
                b.append((label,(startX,startY,endX,endY,xvec,yvec)))
            else:
                b.append((label,(startX,startY,endX,endY,[],[])))
            D[p]=b
    print(f'[INFO] finish loading {len(D)} images from the csv')
    return D

#%%
def main():
    # add command line arguments for config file
    ap=argparse.ArgumentParser()
    ap.add_argument('-c','--config_path',required=True,help='path to config.py')
    args=vars(ap.parse_args())
    config_path=args['config_path']

    config_name=os.path.splitext(os.path.split(config_path)[1])[0]
    spec = importlib.util.spec_from_file_location(config_name,config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    # Catch errors for unspecified config parameters
    try:
        MASK_OPTION=config.MASK_OPTION
    except:
        print('[INFO] Faster R-CNN by default since MASK_OPTION is not defined in ',args['config_path'])
        MASK_OPTION=False   
    try:
        RESIZE_OPTION=config.RESIZE_OPTION
    except:
        print('[INFO] No resizing because RESIZE_OPTION is not defined in ',args['config_path'])
        RESIZE_OPTION=False


    # get class map
    CLASSES = config.CLASSES
    # start counting at 1, class 0 reserved for background
    min_id = min(CLASSES.values())
    if min_id == 0:
        print('[WARNING] Found the smallest class id = 0, increase each class id by one')
        for c in CLASSES:
            CLASSES[c] += 1
                
    #write to the class file 
    if not os.path.isdir(config.OUT_PATH):
        print('[INFO] Not found output path, so create one')
        os.makedirs(config.OUT_PATH)
    with open(config.CLASSES_FILE,'w') as f:
        for (k,v) in CLASSES.items():
            # construct the class information and write to file
            item = ("item {\n"
                "\tid: " + str(v) + "\n"
                "\tname: '" + k + "'\n"
                "}\n")
            f.write(item)

    # load annotations
    D_train = load_csv(config.TRAIN_PATH, config.TRAIN_ANNOT_PATH, CLASSES, MASK_OPTION)
    D_test = load_csv(config.TEST_PATH, config.TEST_ANNOT_PATH, CLASSES, MASK_OPTION)
    
    # initialize datasplit files:
    datasets=[
        ('train', D_train, config.TRAIN_RECORD),
        ('test', D_test, config.TEST_RECORD)
        ]
    
    # build tensorflow record files
    # loop over training and testing splits
    for (dType,D,outputPath) in datasets:
        # initialize the TensorFlow writer and total_imgs #of examples written to file
        print('[INFO] processing "{}"...'.format(dType))
        writer=tf.io.TFRecordWriter(outputPath)
        total_imgs=0
        total_bboxs = 0
        # loop over all keys in the current set

        for k in D.keys():
            img=cv2.imread(k)
            (h0,w0)=img.shape[:2]
            # resize input image
            if RESIZE_OPTION:
                try:
                    MAX_W=config.MAX_W
                    if w0>MAX_W:
                        img_resize=resize(img,width=MAX_W)
                        (h,w)=img_resize.shape[:2]
                        print(f'[INFO] Resized input image & polygons to image size w={w}, h={h}.')
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
            if MASK_OPTION:
                tfAnnot.is_mask=True

            # loop over image bounding boxes + labels
            for (label,(startX,startY,endX,endY,xvec,yvec)) in D[k]:

                # TensorFlow requires normalized bounding boxes
                # Normalized bounding boxes don't need resizing
                xMin=startX/w0
                xMax=endX/w0
                yMin=startY/h0
                yMax=endY/h0

                # update bounding boxes and label lists
                tfAnnot.xMins.append(xMin)
                tfAnnot.xMaxs.append(xMax)
                tfAnnot.yMins.append(yMin)
                tfAnnot.yMaxs.append(yMax)
                tfAnnot.textLabels.append(label.encode('utf8'))
                tfAnnot.classes.append(CLASSES[label])
                # tfAnnot.difficult.append(0)
                
                if MASK_OPTION:
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
                total_bboxs += 1        
            total_imgs+=1

            # encode the data point attributes using the TensorFlow helper functions
            features=tf.train.Features(feature=tfAnnot.build())
            example=tf.train.Example(features=features)

            # add the example to the writer
            writer.write(example.SerializeToString())
        # close writer adn show diagnostics
        writer.close()
        print(f'[INFO] {total_imgs} images with {total_bboxs} bboxs saved for "{dType}"')

# check for main
if __name__=='__main__':
    main()
