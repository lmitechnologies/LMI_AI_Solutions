#%% import modules
import cv2
from object_detection.utils import label_map_util
import tensorflow as tf
import numpy as np 
import argparse
import image_utils.Rosenbrock as imutils
import ast
import os
import glob
import time
import csv
from tensorflow.python.saved_model.signature_constants import \
    DEFAULT_SERVING_SIGNATURE_DEF_KEY


# TODO:
# conform to if __name__==main() convention
# check for improvements

#%% get arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--savedmodel", required=True, help="path to saved model")
ap.add_argument("-l", "--labels", required=True,help="labels file")
ap.add_argument("-i", "--image", required=True,help="path to input image, or directory of png files.")
ap.add_argument("--min_dim",required=False,default=600,help="image width (typically match to training)")
ap.add_argument("--max_dim",required=False,default=1024,help="image width (typically match to training)")
ap.add_argument("-n", "--num-classes", type=int, required=True,help="# of class labels")
ap.add_argument("-c", "--min-confidence", type=float, default=0.5,help="minimum probability used to filter weak detections")
ap.add_argument("-d","--draw",default='True',help='draw the results on the input image')
ap.add_argument("-s","--save",default=None,help='save path')
ap.add_argument("-o","--csv-output",default=None,help='write to csv')
ap.add_argument("--gpu_memory_limit",type=int,default=0,help='gpu memory limit')
# add TensorRT Option
ap.add_argument("--trt_option",dest="use_trt",action='store_true')
ap.add_argument("--no_trt_option",dest="use_trt",action='store_false')
ap.set_defaults(use_trt=False)

args = vars(ap.parse_args())

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    if args['gpu_memory_limit']==0:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
            print('[INFO] Using GPU memory growth.')
        except RuntimeError as e:
            print(e)
    else:
        try:
            tf.config.experimental.set_virtual_device_configuration( \
            gpus[0],\
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=args['gpu_memory_limit'])])
            print(f'[INFO] Using GPU memory limit = {args["gpu_memory_limit"]}.')
        except RuntimeError as e:  
            print(e)
    # Virtual devices must be set before GPUs have been initialized
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    

def _force_gpu_resync(func):
    p = tf.constant(0.)  # Create small tensor to force GPU resync
    def wrapper(*args, **kwargs):
        rslt = func(*args, **kwargs)
        (p + 1.).numpy()  # Sync the GPU
        return rslt
    return wrapper


#%% get colors for labels
class_max=args['num_classes']
#COLORS=np.random.uniform(0,255,size=(class_max,3))
COLORS=[(0,0,255),(255,0,0),(0,255,0),(102,51,153),(255,140,0),(105,105,105),(127,25,27),(9,200,100)]

#%% Load the model

detect_fn=tf.saved_model.load(args['savedmodel'])
if args['use_trt']:
    print('[INFO] Using TRT')
    detect_fn=detect_fn.signatures['serving_default']


@_force_gpu_resync
@tf.function(jit_compile=False)
def predict(image):
    return detect_fn(image)

#%% load class labels
labelMap=label_map_util.load_labelmap(args['labels'])
categories=label_map_util.convert_label_map_to_categories(labelMap,max_num_classes=class_max,use_display_name=True)
categoryIdx=label_map_util.create_category_index(categories)
 
csv_path=args['csv_output']
if csv_path is not None:
    csv_out=[]


#%% create a tf session to perform inference


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
    # imageTensor=model.get_tensor_by_name('image_tensor:0')
    # boxesTensor=model.get_tensor_by_name('detection_boxes:0')
    # try:
    #     masksTensor=model.get_tensor_by_name('detection_masks:0')
    # except:
    #     masksTensor=None
    #     print('[INFO] No masks found.  Proceeding with Faster R-CNN.')

    # #get score for class lable
    # scoresTensor=model.get_tensor_by_name('detection_scores:0')
    # classesTensor=model.get_tensor_by_name('detection_classes:0')
    # numDetections=model.get_tensor_by_name('num_detections:0')

    print('[INFO] Process image:',image_file)
    #get image from disk
    image=cv2.imread(image_file)
    (H0,W0)=image.shape[:2]
    
    print('[INFO] Image height: %5d, Image width: %5d' % (H0,W0))
    #resize image to match training shape

    dmin=int(args['min_dim'])
    dmax=int(args['max_dim'])

    if W0<=H0:
        if W0 != dmin:
            image=imutils.resize(image,width=dmin)
            (H,W)=image.shape[:2]
            print('[INFO] Resizing image, height: %5d, width: %5d' % (H,W) )
        else:
            W,H=W0,H0
        if H>dmax:
            image=imutils.resize(image,height=dmax)
            (H,W)=image.shape[:2]
            print('[INFO] Resizing image, height: %5d, width: %5d' % (H,W) )
    else:
        if H0 != dmin:
            image=imutils.resize(image,height=dmin)
            (H,W)=image.shape[:2]
            print('[INFO] Resizing image, height: %5d, width: %5d' % (H,W) )
        else:
            W,H=W0,H0
        if W>dmax:
            image=imutils.resize(image,width=dmax)
            (H,W)=image.shape[:2]
            print('[INFO] Resizing image, height: %5d, width: %5d' % (H,W) )

    output=image.copy()
    image=cv2.cvtColor(image.copy(),cv2.COLOR_BGR2RGB)
    image=np.expand_dims(image,axis=0)
    image_tensor=tf.convert_to_tensor(image)
    

    start=time.time()
    detections = predict(image_tensor)
    end=time.time()
    proc_time[i]=end-start

    boxes=detections['detection_boxes'].numpy()
    scores=detections['detection_scores'].numpy()
    labels=detections['detection_classes'].numpy()
    N=detections['num_detections'].numpy()
    # masks=np.zeros(boxes.shape)
    if 'mask_predictions' in list(detections.keys()):
        masks=detections['detection_masks'].numpy()
    else:
        masks=np.empty((boxes.shape[1],1))
        masks[:]=np.nan
    
    #perform inference and compute bounding boxes, probabilities, and labels
    
    #skip masks if Faster R-CNN
    # if masksTensor is not None:
    #     (boxes,masks,scores,labels,N)=sess.run([boxesTensor,masksTensor,scoresTensor,classesTensor,numDetections],feed_dict={imageTensor:image})
    # else:
    #     (boxes,scores,labels,N)=sess.run([boxesTensor,scoresTensor,classesTensor,numDetections],feed_dict={imageTensor:image})
    #     masks=np.zeros(boxes.shape)

        
    
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

        # box=np.squeeze(box)
        # mask=np.squeeze(mask)
        # score=np.squeeze(score)
        # label=np.squeeze(label)

        #extract bounding box
        (startY,startX,endY,endX)=box

        #add each new bounding box to the csv write buffer
        label=categoryIdx[int(label)]
        if csv_path is not None:
            print('Found Object:',label['name'])
            #scale bounding box from [0,1] to [W0,H0]
            startX0=int(startX*W0)
            startY0=int(startY*H0)
            endX0=int(endX*W0)
            endY0=int(endY*H0)
            csv_out.append([os.path.split(image_file)[1],label['name'],'rect','upper left',startX0,startY0])
            csv_out.append([os.path.split(image_file)[1],label['name'],'rect','lower right',endX0,endY0])
            if not np.isnan(mask).any():
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
            if not np.isnan(mask).any():
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
            cv2.rectangle(output, (startX, startY), (endX, endY),COLORS[idx], 1)
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

if len(proc_time)>3:
    proc_time=np.sort(proc_time)
    print('\n')
    print('[INFO] Average runtime: ',np.mean(proc_time[1:-1]))
    print('[INFO] Min runtime: ',np.min(proc_time[1:-1]))
    print('[INFO] Max runtime: ',np.max(proc_time[1:-1]))

# initialize csv output file
if csv_path is not None:
    with open(csv_path,'w',newline='') as csvfile:
        labelWriter=csv.writer(csvfile,delimiter=';')
        for row in csv_out:
            labelWriter.writerow(row)    
