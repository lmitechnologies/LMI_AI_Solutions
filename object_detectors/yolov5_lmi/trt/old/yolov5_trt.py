import random
import time
import cv2
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda

from yolov5_lmi.trt.old.base_trt_model import TRT_Model


class YoLov5TRT(TRT_Model):
    """
    description: A YOLOv5 class that warps TensorRT ops, preprocess and postprocess ops.
    """

    def __init__(self, engine_file_path, plugin_path, class_map, conf_thres, iou_thres=0.4, max_output_bbox_count=1000):
        super().__init__(engine_file_path,plugin_path)
        #load configs
        self.class_map = class_map
        self.conf_thres = conf_thres
        self.nms_iou_thres = iou_thres
        self.max_output_bbox_count = max_output_bbox_count
        
        colors = [(0,0,255),(255,0,0),(0,255,0),(102,51,153),(255,140,0),(105,105,105),(127,25,27),(9,200,100)]
        #<cls name: (R,G,B)>
        self.color_map = {}
        min_id = min(class_map.keys())
        for i in class_map:
            cls_name = class_map[i]
            if min_id != 0:
                i -= min_id
            if i < len(colors):
                self.color_map[cls_name] = colors[i]
            else:
                self.color_map[cls_name] = tuple([random.randint(0,255) for _ in range(3)])
        
        
    def infer(self, images_raw:list, conf_thres:dict={}, nms_iou_thres=None):
        start = time.time()
        # loading default thresholds
        if not conf_thres:
            conf_thres = self.conf_thres
        if nms_iou_thres is None:
            nms_iou_thres = self.nms_iou_thres
        # Make self the active context, pushing it on top of the context stack.
        self.ctx.push()
        errors = []
        # Do image preprocess
        batch_input_images = np.empty(shape=[self.batch_max_size, 3, self.input_h, self.input_w])
        batch_size = len(images_raw)
        for i, image in enumerate(images_raw):
            input_image,raw_image,errs = self.preprocess_image(image)
            errors += errs
            np.copyto(batch_input_images[i], input_image)
        batch_input_images = np.ascontiguousarray(batch_input_images)
        preproc_time = time.time() - start

        start = time.time()
        # Copy input image to host buffer
        np.copyto(self.host_inputs[0], batch_input_images.ravel())
        # Transfer input data  to the GPU.
        cuda.memcpy_htod_async(self.cuda_inputs[0], self.host_inputs[0], self.stream)
        # Run inference.
        self.context.execute_async(batch_size=batch_size, bindings=self.bindings, stream_handle=self.stream.handle)
        # Transfer predictions back from the GPU.
        cuda.memcpy_dtoh_async(self.host_outputs['prob'], self.cuda_outputs['prob'], self.stream)
        # Synchronize the stream
        self.stream.synchronize()
        exec_time = time.time() - start

        start = time.time()
        # Remove any context from the top of the context stack, deactivating it.
        self.ctx.pop()
        # Here we use the first row of output in that batch_size = 1
        output = self.host_outputs['prob']
        # Do postprocess
        results = []
        for i in range(batch_size):
            result_boxes, result_scores, result_classid = self.post_process(
                output[i * (6*self.max_output_bbox_count + 1): (i + 1) * (6*self.max_output_bbox_count + 1)], self.input_h, self.input_w, conf_thres, nms_iou_thres
            )
            pred_classes = [self.class_map[int(cid)]  for cid in result_classid]
            results.append({'boxes':result_boxes.tolist(), 'scores':result_scores.tolist(), 'classes':pred_classes})
        postproc_time = time.time() - start
        return results, {'pre':preproc_time, 'exec':exec_time, 'post':postproc_time}, errors


    def xywh2xyxy(self, origin_h, origin_w, x):
        """
        description:    Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        param:
            origin_h:   height of original image
            origin_w:   width of original image
            x:          A boxes numpy, each row is a box [center_x, center_y, w, h]
        return:
            y:          A boxes numpy, each row is a box [x1, y1, x2, y2]
        """
        y = np.zeros_like(x)
        r_w = self.input_w / origin_w
        r_h = self.input_h / origin_h
        if r_h > r_w:
            y[:, 0] = x[:, 0] - x[:, 2] / 2
            y[:, 2] = x[:, 0] + x[:, 2] / 2
            y[:, 1] = x[:, 1] - x[:, 3] / 2 - (self.input_h - r_w * origin_h) / 2
            y[:, 3] = x[:, 1] + x[:, 3] / 2 - (self.input_h - r_w * origin_h) / 2
            y /= r_w
        else:
            y[:, 0] = x[:, 0] - x[:, 2] / 2 - (self.input_w - r_h * origin_w) / 2
            y[:, 2] = x[:, 0] + x[:, 2] / 2 - (self.input_w - r_h * origin_w) / 2
            y[:, 1] = x[:, 1] - x[:, 3] / 2
            y[:, 3] = x[:, 1] + x[:, 3] / 2
            y /= r_h

        return y
    

    def post_process(self, output, origin_h, origin_w, conf_thres, nms_iou_thres):
        """
        description: postprocess the prediction
        param:
            output:     A numpy likes [num_boxes,cx,cy,w,h,conf,cls_id, cx,cy,w,h,conf,cls_id, ...] 
            origin_h:   height of original image
            origin_w:   width of original image
            conf_thres(dict): a map of <class_name, threshold>
            nms_iou_thres: non maximum suppresion iou threshold
        return:
            result_boxes: finally boxes, a boxes numpy, each row is a box [x1, y1, x2, y2]
            result_scores: finally scores, a numpy, each element is the score correspoing to box
            result_classid: finally classid, a numpy, each element is the classid correspoing to box
        """
        # Get the num of boxes detected
        num = int(output[0])
        # Reshape to a two dimentional ndarray
        pred = np.reshape(output[1:], (-1, 6))[:num, :]
        # Do nms
        boxes = self.non_max_suppression(pred, origin_h, origin_w, conf_thres, nms_iou_thres)
        result_boxes = boxes[:, :4] if len(boxes) else np.array([])
        result_scores = boxes[:, 4] if len(boxes) else np.array([])
        result_classid = boxes[:, 5] if len(boxes) else np.array([])
        
        return result_boxes, result_scores, result_classid


    def bbox_iou(self, box1, box2, x1y1x2y2=True):
        """
        description: compute the IoU of two bounding boxes
        param:
            box1: A box coordinate (can be (x1, y1, x2, y2) or (x, y, w, h))
            box2: A box coordinate (can be (x1, y1, x2, y2) or (x, y, w, h))            
            x1y1x2y2: select the coordinate format
        return:
            iou: computed iou
        """
        if not x1y1x2y2:
            # Transform from center and width to exact coordinates
            b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
            b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
            b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
            b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
        else:
            # Get the coordinates of bounding boxes
            b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
            b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

        # Get the coordinates of the intersection rectangle
        inter_rect_x1 = np.maximum(b1_x1, b2_x1)
        inter_rect_y1 = np.maximum(b1_y1, b2_y1)
        inter_rect_x2 = np.minimum(b1_x2, b2_x2)
        inter_rect_y2 = np.minimum(b1_y2, b2_y2)
        # Intersection area
        inter_area = np.clip(inter_rect_x2 - inter_rect_x1 + 1, 0, None) * \
                     np.clip(inter_rect_y2 - inter_rect_y1 + 1, 0, None)
        # Union Area
        b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
        b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

        iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

        return iou


    def non_max_suppression(self, prediction, origin_h, origin_w, conf_thres, nms_thres=0.4):
        """
        description: Removes detections with lower object confidence score than 'conf_thres' and performs
        Non-Maximum Suppression to further filter detections.
        param:
            prediction: detections, (x1, y1, x2, y2, conf, cls_id)
            origin_h: original image height
            origin_w: original image width
            conf_thres(dict): a map of <class_name, threshold> to filter detections
            nms_thres: a iou threshold to filter detections
        return:
            boxes: output after nms with the shape (x1, y1, x2, y2, conf, cls_id)
        """
        # Get the boxes that score > CONF_THRESH
        M = np.zeros(prediction.shape[0], dtype=bool)
        for i,pred in enumerate(prediction):
            name = self.class_map[pred[5]]
            M[i] = pred[4] >= conf_thres[name]
        boxes = prediction[M]
        # Trandform bbox from [center_x, center_y, w, h] to [x1, y1, x2, y2]
        boxes[:, :4] = self.xywh2xyxy(origin_h, origin_w, boxes[:, :4])
        # clip the coordinates
        boxes[:, 0] = np.clip(boxes[:, 0], 0, origin_w -1)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, origin_w -1)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, origin_h -1)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, origin_h -1)
        # Object confidence
        confs = boxes[:, 4]
        # Sort by the confs
        boxes = boxes[np.argsort(-confs)]
        # Perform non-maximum suppression
        keep_boxes = []
        while boxes.shape[0]:
            large_overlap = self.bbox_iou(np.expand_dims(boxes[0, :4], 0), boxes[:, :4]) > nms_thres
            label_match = boxes[0, -1] == boxes[:, -1]
            # Indices of boxes with lower confidence scores, large IOUs and matching labels
            invalid = large_overlap & label_match
            keep_boxes += [boxes[0]]
            boxes = boxes[~invalid]
        boxes = np.stack(keep_boxes, 0) if len(keep_boxes) else np.array([])
        return boxes
