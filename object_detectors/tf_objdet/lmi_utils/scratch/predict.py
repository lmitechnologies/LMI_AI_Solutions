#%% import modules
from object_detection.utils import label_map_util
import tensorflow as tf
import numpy as np 
import argparse
import imutils
import cv2
import ast
import os
import glob
import time
import csv

tf_config = tf.compat.v1.ConfigProto()
# tf_config.gpu_options.per_process_gpu_memory_fraction = 0.9
# tf_config.gpu_options.allow_growth = True

#%% get arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True, help="base path for frozen checkpoint detection graph")
ap.add_argument("-l", "--labels", required=True,help="labels file")
ap.add_argument("-i", "--image", required=True,help="path to input image, or directory of png files.")
ap.add_argument("--min_dim",required=False,default=600,help="image width (typically match to training)")
ap.add_argument("--max_dim",required=False,default=1024,help="image width (typically match to training)")
ap.add_argument("-n", "--num-classes", type=int, required=True,help="# of class labels")
ap.add_argument("-c", "--min-confidence", type=float, default=0.5,help="minimum probability used to filter weak detections")
ap.add_argument("-d","--draw",default='True',help='draw the results on the input image')
ap.add_argument("-s","--save",default=None,help='save path')
ap.add_argument("-o","--csv-output",default=None,help='write to csv')

args = vars(ap.parse_args())

#%% get colors for labels
class_max=args['num_classes']
#COLORS=np.random.uniform(0,255,size=(class_max,3))
COLORS=[(0,0,255),(255,0,0),(0,255,0)]

#%% Load the model
model=tf.Graph()
with model.as_default():
    # initialize the graph definition
    graphDef=tf.GraphDef()

    #load graph from disk
    with tf.gfile.GFile(args['model'],'rb') as f:
        serializedGraph=f.read()
        graphDef.ParseFromString(serializedGraph)
        tf.import_graph_def(graphDef,name='')

#%% load class labels
labelMap=label_map_util.load_labelmap(args['labels'])
categories=label_map_util.convert_label_map_to_categories(labelMap,max_num_classes=class_max,use_display_name=True)
categoryIdx=label_map_util.create_category_index(categories)
 
csv_path=args['csv_output']
if csv_path is not None:
    csv_out=[['Path','Label']]

#%% create a tf session to perform inference
with model.as_default():
    with tf.Session(graph=model) as sess:
        image_path=args['image']
        if os.path.splitext(image_path)[-1]=='':
            image_path=image_path+'/*.png'
            images=glob.glob(image_path)
            single_image=False
        else:
            images=[image_path]
            single_image=True

        proc_time=np.zeros(len(images))
        for i,image_file in enumerate(images):
            #get input image and bounding box
            imageTensor=model.get_tensor_by_name('image_tensor:0')
            boxesTensor=model.get_tensor_by_name('detection_boxes:0')
            try:
                masksTensor=model.get_tensor_by_name('detection_masks:0')
            except:
                masksTensor=None
                print('[INFO] No masks found.  Proceeding with Faster R-CNN.')

            #get score for class lable
            scoresTensor=model.get_tensor_by_name('detection_scores:0')
            classesTensor=model.get_tensor_by_name('detection_classes:0')
            numDetections=model.get_tensor_by_name('num_detections:0')

            print('[INFO] Process image:',image_file)
            #get image from disk
            image=cv2.imread(image_file)
            (H0,W0)=image.shape[:2]
            
            print('[INFO] Image height: %5d, Image width: %5d' % (H0,W0))
            #resize image to match training shape

            dmin=int(args['min_dim'])
            dmax=int(args['max_dim'])

            if W0<=H0:
                image=imutils.resize(image,width=dmin)
                (H,W)=image.shape[:2]
                print('[INFO] Resizing image, height: %5d, width: %5d' % (H,W) )
                if H>dmax:
                    image=imutils.resize(image,height=dmax)
                    (H,W)=image.shape[:2]
                    print('[INFO] Resizing image, height: %5d, width: %5d' % (H,W) )
            else:
                image=imutils.resize(image,height=dmin)
                (H,W)=image.shape[:2]
                print('[INFO] Resizing image, height: %5d, width: %5d' % (H,W) )
                if W>dmax:
                    image=imutils.resize(image,width=dmax)
                    (H,W)=image.shape[:2]
                    print('[INFO] Resizing image, height: %5d, width: %5d' % (H,W) )

            output=image.copy()
            image=cv2.cvtColor(image.copy(),cv2.COLOR_BGR2RGB)
            image=np.expand_dims(image,axis=0)
            
            #perform inference and compute bounding boxes, probabilities, and labels
            start=time.time()
            #skip masks if Faster R-CNN
            if masksTensor is not None:
                (boxes,masks,scores,labels,N)=sess.run([boxesTensor,masksTensor,scoresTensor,classesTensor,numDetections],feed_dict={imageTensor:image})
            else:
                (boxes,scores,labels,N)=sess.run([boxesTensor,scoresTensor,classesTensor,numDetections],feed_dict={imageTensor:image})
                masks=np.zeros(boxes.shape)
                
            end=time.time()
            proc_time[i]=end-start
            print('[INFO] Processing Time: ',proc_time[i], 's.')
            #recast to 1-D array
            boxes=np.squeeze(boxes)
            masks=np.squeeze(masks)
            scores=np.squeeze(scores)
            labels=np.squeeze(labels)

            #loop over bounding box predictions
            draw=ast.literal_eval(args['draw'])
            for (box,mask,score,label) in zip(boxes,masks,scores,labels):
                if score<args['min_confidence']:
                    continue

                #extract bounding box
                (startY,startX,endY,endX)=box

                #add each new bounding box to the csv write buffer
                label=categoryIdx[label]
                if csv_path is not None:
                    print('Found Object:',label['name'])
                    #scale bounding box from [0,1] to [W0,H0]
                    startX0=int(startX*W0)
                    startY0=int(startY*H0)
                    endX0=int(endX*W0)
                    endY0=int(endY*H0)
                    csv_out.append([os.path.split(image_file)[1],label['name'],'rect','upper left',startX0,startY0])
                    csv_out.append([os.path.split(image_file)[1],label['name'],'rect','lower right',endX0,endY0])
                    if masksTensor is not None:
                        mask0=cv2.resize(mask,(endX0-startX0,endY0-startY0),interpolation=cv2.INTER_CUBIC)
                        mask0 = (mask0>args['min_confidence'])
                        canvas=np.zeros((H0,W0),dtype=np.uint8)
                        canvas[startY0:endY0, startX0:endX0][mask0] = np.uint8(255)
                        contours,_=cv2.findContours(canvas,mode=cv2.RETR_EXTERNAL,method=cv2.CHAIN_APPROX_SIMPLE)
                        for contour in contours:
                            cont_size=contour.shape[0]
                            xval=list(contour.reshape((cont_size,2))[:,0])
                            yval=list(contour.reshape((cont_size,2))[:,1])
                            csv_out.append([os.path.split(image_file)[1],label['name'],'polygon','x values'] + xval)
                            csv_out.append([os.path.split(image_file)[1],label['name'],'polygon','y values'] + yval)

                # draw the prediction on the output image 1 box at a time
                if (draw==True) or (args['save'] is not None):
                    #scale bounding box from [0,1] to [W,H]
                    startX=int(startX*W)
                    startY=int(startY*H)
                    endX=int(endX*W)
                    endY=int(endY*H)

                    idx = int(label["id"]) - 1

                    #scale the mask
                    if masksTensor is not None:
                        # fit objectness mask to bounding box
                        mask=cv2.resize(mask,(endX-startX,endY-startY),interpolation=cv2.INTER_CUBIC)
                        # keep high confidence pixels
                        mask = (mask>args['min_confidence'])
                        # extract roi in image coordinates
                        roi=output[startY:endY,startX:endX]
                        # get only pixels with objects
                        roi=roi[mask]
                        color_array=np.array(COLORS[idx],dtype=np.float32)
                        blended = ((0.4 * color_array) + (0.6 * roi)).astype("uint8")
                        # set pixels in image
                        output[startY:endY, startX:endX][mask] = blended
                    else:
                        pass
                    
                    label = "{}: {:.2f}".format(label["name"], score)
                    cv2.rectangle(output, (startX, startY), (endX, endY),COLORS[idx], 2)
                    y = startY - 10 if startY - 10 > 10 else startY + 10
                    cv2.putText(output, label, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.3, COLORS[idx], 1)
                    # show the output image, 1 found image at a time

                    if draw==True:
                        cv2.imshow("Output", output)
                        if single_image==True:
                            cv2.waitKey(0)
                        elif single_image==False:
                            cv2.waitKey(500)

            # save the output image with all boxes        
            if args['save'] is not None:
                splitpath=os.path.split(image_file)
                out_filename=os.path.splitext(splitpath[1])[0]+'_validation.png'
                out_path=args['save']+'/'+out_filename
                print('[INFO] saving file:',out_path)
                cv2.imwrite(out_path,output)
        
        print('[INFO] Average runtime: ',np.mean(proc_time))
        print('[INFO] Min runtime: ',np.min(proc_time))
        print('[INFO] Max runtime: ',np.max(proc_time))

        # initialize csv output file
        if csv_path is not None:
            with open(csv_path,'w',newline='') as csvfile:
                labelWriter=csv.writer(csvfile,delimiter=';')
                for row in csv_out:
                    labelWriter.writerow(row)    
