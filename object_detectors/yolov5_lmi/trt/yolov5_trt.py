from collections import OrderedDict, namedtuple
import cv2
import logging
import os
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import tensorrt as trt

# from ..utils.general import xywh2xyxy, clip_boxes, scale_boxes, clip_segments, scale_segments
# from ..utils.metrics import box_iou
# from ..utils.segment.general import masks2segments, process_mask, crop_mask


class YoLov5TRT:
    logger = logging.getLogger('ENGINE')
    
    
    def __init__(self, engines) -> None:
        """
        args:
            engines(str or a list): the path to the tensorRT engine file
        """
        w = str(engines[0] if isinstance(engines, list) else engines)
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
        else:
            self.logger.warning('Cannot find GPU device. Use CPU instead.')
            device = torch.device('cpu')

        self.logger.info(f'Loading {w} for TensorRT inference...')
        Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
        trt_logger = trt.Logger(trt.Logger.INFO)
        with open(w, 'rb') as f, trt.Runtime(trt_logger) as runtime:
            model = runtime.deserialize_cuda_engine(f.read())
        context = model.create_execution_context()
        bindings = OrderedDict()
        output_names = []
        fp16 = False  # default updated below
        dynamic = False
        for i in range(model.num_bindings):
            name = model.get_binding_name(i)
            dtype = trt.nptype(model.get_binding_dtype(i))
            
            self.logger.info(f'binding {name} with the shape of {model.get_binding_shape(i)}')
            if model.binding_is_input(i):
                input_h,input_w = model.get_binding_shape(i)[-2:]
                self.logger.info(f'found tensorRT input h,w = {input_h,input_w}')
                if -1 in tuple(model.get_binding_shape(i)):  # dynamic
                    dynamic = True
                    context.set_binding_shape(i, tuple(model.get_profile_shape(0, i)[2]))
                if dtype == np.float16:
                    fp16 = True
            else:  # output
                output_names.append(name)
            shape = tuple(context.get_binding_shape(i))
            im = torch.from_numpy(np.empty(shape, dtype=dtype)).to(device)
            bindings[name] = Binding(name, dtype, shape, im, int(im.data_ptr()))
        binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())
        batch_size = bindings['images'].shape[0]
        if len(output_names)>1:
            use_mask = 1
        else:
            use_mask = 0
        self.__dict__.update(locals())


    def forward(self, im):
        if self.fp16:
            im = im.half()
        if self.dynamic and im.shape != self.bindings['images'].shape:
            i = self.model.get_binding_index('images')
            self.context.set_binding_shape(i, im.shape)  # reshape if dynamic
            self.bindings['images'] = self.bindings['images']._replace(shape=im.shape)
            for name in self.output_names:
                i = self.model.get_binding_index(name)
                self.bindings[name].data.resize_(tuple(self.context.get_tensor_shape(i)))
        s = self.bindings['images'].shape
        assert im.shape == s, f"input size {im.shape} {'>' if self.dynamic else 'not equal to'} max model size {s}"
        self.binding_addrs['images'] = int(im.data_ptr())
        self.context.execute_v2(list(self.binding_addrs.values()))
        y = [self.bindings[x].data for x in sorted(self.output_names)]
        if isinstance(y, (list, tuple)):
            return self.from_numpy(y[0]) if len(y) == 1 else [self.from_numpy(x) for x in y]
        else:
            return self.from_numpy(y)
        
        
    def from_numpy(self, x):
        return torch.from_numpy(x).to(self.device) if isinstance(x, np.ndarray) else x
    
    
    def warmup(self, imgsz=(1, 3, 640, 640)):
        """
        warm up the model once
        Args:
            imgsz (tuple, optional): NCHW format. Defaults to (1, 3, 640, 640).
            
        """
        # Warmup model by running inference once
        im = torch.empty(*imgsz, dtype=torch.half if self.fp16 else torch.float, device=self.device)  # input
        return self.forward(im)
    
    
    def preprocess(self, im0, BGR_to_RGB=False):
        """im preprocess
            BGR_to_RGB -> normalization -> CHW -> contiguous -> BCHW
        Args:
            im0 (numpy.ndarray): the input numpy array, HWC format
            BGR_to_RGB (boolean): change im0 to RGB
        """
        if BGR_to_RGB:
            im0 = im0[:,:,::-1]
        im = im0.astype(np.float32)
        im /= 255 # normalize to [0,1]
        im = im.transpose((2, 0, 1)) # HWC to CHW
        im = np.ascontiguousarray(im)  # contiguous
        if len(im.shape) == 3:
            im = im[None] # expand for batch dim
        return self.from_numpy(im), im0
    
    
    def load_with_preprocess(self, im_path:str):
        """im preprocess

        Args:
            im_path (str): the path to the image, could be either .npy, .png, or other image formats
            
        """
        ext = os.path.splitext(im_path)[-1]
        if ext=='.npy':
            im0 = np.load(im_path)
        else:
            im0 = cv2.imread(im_path) #BGR format
            im0 = im0[:,:,::-1] #BGR to RGB
        return self.preprocess(im0)
    
    
    def postprocess(self,prediction,im0,conf_thres,proto=None,iou_thres=0.45,agnostic=False,max_det=100):
        """
        Args:
            prediction (list): a list of object detection predictions
            im0 (np.ndarray): the numpy image
            conf_thres (dict): confidence threshold dict <class_id: confidence>.
            proto (tensor, optional): mask predictions. Defaults to None.
            iou_thres (float, optional): iou threshold. Defaults to 0.45.
            max_det (int, optional): the max number of detections. Defaults to 100.
            agnostic (bool, optional): perform class-agnostic NMS. Defaults to False.
        """
        # Process predictions
        pred = self.non_max_suppression(prediction,conf_thres,iou_thres,agnostic=agnostic,max_det=max_det)
        segments = []
        masks = []
        for i,det in enumerate(pred):  # per image
            if len(det)==0:
                continue
            # Rescale boxes from img_size to im0 size
            model_shape = (self.input_h,self.input_w)
            if proto is not None:
                mask = self.process_mask(proto[i], det[:, 6:], det[:, :4], model_shape, upsample=True) 
                masks += [mask]
                segs = [
                        self.scale_segments(model_shape, x, im0.shape, normalize=False)
                        for x in reversed(self.masks2segments(mask))]
                segments += [segs]
            det[:, :4] = self.scale_boxes(model_shape, det[:, :4], im0.shape).round()
            
        return [pred, segments, masks] if self.use_mask else pred
    
    
    def non_max_suppression(self, 
        prediction,
        conf_thres,
        iou_thres=0.45,
        classes=None,
        agnostic=False,
        labels=(),
        max_det=100,
        nm=0,  # number of masks
        ):
        """Non-Maximum Suppression (NMS) on inference results to reject overlapping detections
        Returns:
            list of detections, on (n,6) tensor per image [xyxy, conf, cls]
        """

        if isinstance(prediction, (list, tuple)):  # YOLOv5 model in validation model, output = (inference_out, loss_out)
            prediction = prediction[0]  # select only inference output
            
        if self.use_mask:
            nm = 32

        bs = prediction.shape[0]  # batch size
        nc = prediction.shape[2] - nm - 5  # number of classes
        xc = prediction[..., 4] > 0.01   # candidates

        # Checks
        # assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
        assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

        # Settings
        # min_wh = 2  # (pixels) minimum box width and height
        max_wh = 7680  # (pixels) maximum box width and height
        max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
        time_limit = 0.5 + 0.05 * bs  # seconds to quit after
        redundant = True  # require redundant detections
        # multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
        merge = False  # use merge-NMS

        # t = time.time()
        mi = 5 + nc  # mask start index
        output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
        for xi, x in enumerate(prediction):  # image index, image inference
            # Apply constraints
            # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
            x = x[xc[xi]]  # confidence

            # Cat apriori labels if autolabelling
            if labels and len(labels[xi]):
                lb = labels[xi]
                v = torch.zeros((len(lb), nc + nm + 5), device=x.device)
                v[:, :4] = lb[:, 1:5]  # box
                v[:, 4] = 1.0  # conf
                v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
                x = torch.cat((x, v), 0)

            # If none remain process next image
            if not x.shape[0]:
                continue

            # Compute conf
            x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

            # Box/Mask
            box = self.xywh2xyxy(x[:, :4])  # center_x, center_y, width, height) to (x1, y1, x2, y2)
            mask = x[:, mi:]  # zero columns if no masks

            # Detections matrix nx6 (xyxy, conf, cls)
            # if multi_label:
            #     i, j = (x[:, 5:mi] > conf_thres).nonzero(as_tuple=False).T
            #     x = torch.cat((box[i], x[i, 5 + j, None], j[:, None].float(), mask[i]), 1)
            # best class only
            conf, j = x[:, 5:mi].max(1, keepdim=True)
            conf_thres_tmp = torch.tensor([conf_thres[c.item()] for c in j], device=x.device)
            x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres_tmp]

            # Filter by class
            if classes is not None:
                x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

            # Apply finite constraint
            # if not torch.isfinite(x).all():
            #     x = x[torch.isfinite(x).all(1)]

            # Check shape
            n = x.shape[0]  # number of boxes
            if not n:  # no boxes
                continue
            elif n > max_nms:  # excess boxes
                x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence
            else:
                x = x[x[:, 4].argsort(descending=True)]  # sort by confidence

            # Batched NMS
            c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
            boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
            i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
            if i.shape[0] > max_det:  # limit detections
                i = i[:max_det]
            if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
                # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
                iou = self.box_iou(boxes[i], boxes) > iou_thres  # iou matrix
                weights = iou * scores[None]  # box weights
                x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
                if redundant:
                    i = i[iou.sum(1) > 1]  # require redundancy

            output[xi] = x[i]
            # if (time.time() - t) > time_limit:
            #     print(f'WARNING ⚠️ NMS time limit {time_limit:.3f}s exceeded')
            #     break  # time limit exceeded
        return output
    

    def box_iou(self, box1, box2, eps=1e-7):
        # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
        """
        Return intersection-over-union (Jaccard index) of boxes.
        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
        Arguments:
            box1 (Tensor[N, 4])
            box2 (Tensor[M, 4])
        Returns:
            iou (Tensor[N, M]): the NxM matrix containing the pairwise
                IoU values for every element in boxes1 and boxes2
        """

        # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
        (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
        inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)

        # IoU = inter / (area1 + area2 - inter)
        return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)


    def xywh2xyxy(self, x):
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
        y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
        y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
        y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
        return y
    
    
    def clip_boxes(self, boxes, shape):
        # Clip boxes (xyxy) to image shape (height, width)
        if isinstance(boxes, torch.Tensor):  # faster individually
            boxes[..., 0].clamp_(0, shape[1])  # x1
            boxes[..., 1].clamp_(0, shape[0])  # y1
            boxes[..., 2].clamp_(0, shape[1])  # x2
            boxes[..., 3].clamp_(0, shape[0])  # y2
        else:  # np.array (faster grouped)
            boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
            boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2
    
    
    def scale_boxes(self, img1_shape, boxes, img0_shape, ratio_pad=None):
        # Rescale boxes (xyxy) from img1_shape to img0_shape
        if ratio_pad is None:  # calculate from img0_shape
            gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
            pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
        else:
            gain = ratio_pad[0][0]
            pad = ratio_pad[1]

        boxes[..., [0, 2]] -= pad[0]  # x padding
        boxes[..., [1, 3]] -= pad[1]  # y padding
        boxes[..., :4] /= gain
        self.clip_boxes(boxes, img0_shape)
        return boxes
    
    
    def clip_segments(self, segments, shape):
        # Clip segments (xy1,xy2,...) to image shape (height, width)
        if isinstance(segments, torch.Tensor):  # faster individually
            segments[:, 0].clamp_(0, shape[1])  # x
            segments[:, 1].clamp_(0, shape[0])  # y
        else:  # np.array (faster grouped)
            segments[:, 0] = segments[:, 0].clip(0, shape[1])  # x
            segments[:, 1] = segments[:, 1].clip(0, shape[0])  # y
            
            
    def scale_segments(self, img1_shape, segments, img0_shape, ratio_pad=None, normalize=False):
        # Rescale coords (xyxy) from img1_shape to img0_shape
        if ratio_pad is None:  # calculate from img0_shape
            gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
            pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
        else:
            gain = ratio_pad[0][0]
            pad = ratio_pad[1]

        segments[:, 0] -= pad[0]  # x padding
        segments[:, 1] -= pad[1]  # y padding
        segments /= gain
        self.clip_segments(segments, img0_shape)
        if normalize:
            segments[:, 0] /= img0_shape[1]  # width
            segments[:, 1] /= img0_shape[0]  # height
        return segments
    
    
    def crop_mask(self, masks, boxes):
        """
        "Crop" predicted masks by zeroing out everything not in the predicted bbox.
        Vectorized by Chong (thanks Chong).

        Args:
            - masks should be a size [h, w, n] tensor of masks
            - boxes should be a size [n, 4] tensor of bbox coords in relative point form
        """

        n, h, w = masks.shape
        x1, y1, x2, y2 = torch.chunk(boxes[:, :, None], 4, 1)  # x1 shape(1,1,n)
        r = torch.arange(w, device=masks.device, dtype=x1.dtype)[None, None, :]  # rows shape(1,w,1)
        c = torch.arange(h, device=masks.device, dtype=x1.dtype)[None, :, None]  # cols shape(h,1,1)

        return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))
    
    
    def process_mask(self, protos, masks_in, bboxes, shape, upsample=False):
        """
        Crop before upsample.
        proto_out: [mask_dim, mask_h, mask_w]
        out_masks: [n, mask_dim], n is number of masks after nms
        bboxes: [n, 4], n is number of masks after nms
        shape:input_image_size, (h, w)

        return: h, w, n
        """

        c, mh, mw = protos.shape  # CHW
        ih, iw = shape
        masks = (masks_in @ protos.float().view(c, -1)).sigmoid().view(-1, mh, mw)  # CHW

        downsampled_bboxes = bboxes.clone()
        downsampled_bboxes[:, 0] *= mw / iw
        downsampled_bboxes[:, 2] *= mw / iw
        downsampled_bboxes[:, 3] *= mh / ih
        downsampled_bboxes[:, 1] *= mh / ih

        masks = self.crop_mask(masks, downsampled_bboxes)  # CHW
        if upsample:
            masks = F.interpolate(masks[None], shape, mode='bilinear', align_corners=False)[0]  # CHW
        return masks.gt_(0.5)
    
    
    def masks2segments(self, masks, strategy='largest'):
        # Convert masks(n,160,160) into segments(n,xy)
        segments = []
        for x in masks.int().cpu().numpy().astype('uint8'):
            c = cv2.findContours(x, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
            if c:
                if strategy == 'concat':  # concatenate all segments
                    c = np.concatenate([x.reshape(-1, 2) for x in c])
                elif strategy == 'largest':  # select largest segment
                    c = np.array(c[np.array([len(x) for x in c]).argmax()]).reshape(-1, 2)
            else:
                c = np.zeros((0, 2))  # no segments found
            segments.append(c.astype('float32'))
        return segments
    
