import os, sys
import logging 
from collections import OrderedDict, namedtuple
import tensorrt as trt
import torch
import numpy as np
import warnings
import cv2
import glob
import shutil
import time
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('AnomalyModel')

PASS = 'PASS'
FAIL = 'FAIL'

Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))

class AnomalyModel:
    
    def __init__(self, engine_path):
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            logger.warning('GPU device unavailable. Use CPU instead.')
            self.device = torch.device('cpu')
        with open(engine_path, "rb") as f, trt.Runtime(trt.Logger(trt.Logger.WARNING)) as runtime:
            model = runtime.deserialize_cuda_engine(f.read())
        self.context = model.create_execution_context()
        self.bindings = OrderedDict()
        self.output_names = []
        self.fp16 = False
        for i in range(model.num_bindings):
            name = model.get_tensor_name(i)
            dtype = trt.nptype(model.get_tensor_dtype(name))
            shape = tuple(self.context.get_tensor_shape(name))
            logger.info(f'binding {name} ({dtype}) with shape {shape}')
            if model.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                if dtype == np.float16:
                    self.fp16 = True
            else:
                self.output_names.append(name)
            im = self.from_numpy(np.empty(shape, dtype=dtype)).to(self.device)
            self.bindings[name] = Binding(name, dtype, shape, im, int(im.data_ptr()))
        self.binding_addrs = OrderedDict((n, d.ptr) for n, d in self.bindings.items())

    def preprocess(self, image):
        img = cv2.resize(AnomalyModel.normalize(image), self.bindings['input'].shape[-2:], interpolation=cv2.INTER_AREA)
        input_dtype = np.float16 if self.fp16 else np.float32
        input_batch = np.array(np.repeat(np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0), 1, axis=0), dtype=input_dtype)
        return self.from_numpy(input_batch)

    def warmup(self):
        logger.info("warmup started")
        t0 = time.time()
        shape = [i*2 for i in self.bindings['input'].shape[-2:]]+[3,]
        self.predict(np.zeros(shape))
        logger.info(f"warmup ended - {time.time() - t0}")

    def predict(self, image):
        input_batch = self.preprocess(image)
        self.binding_addrs['input'] = int(input_batch.data_ptr())
        self.context.execute_v2(list(self.binding_addrs.values()))
        outputs = {x:self.bindings[x].data.cpu().numpy() for x in self.output_names}
        return outputs['output']

            
    def processContours(self, heatMap, err_dist, color_threshold, size_fail, size_ignore=None):
        """
        DESCRIPTION:
            processContours
                - returns decision based off of pass/fail criteria on contour size or data extracted from specific contour regions
        ARGUMENTS:
            heatMap:    
                    - float32 residual_bgr image (heatmap)
                    - passed in to generate contours
                    - turned into grayscale, colors are inverted (so the anomaly areas are selected for contour detection,
                    and then turned into a binary image... excess large white spots are filled in from fill_start to prevent false-positive contours
                    - resulting image is then passed into findContours, which returns a list of contours
            err_dist: 
                    - numpy array (usually the anomaly_map variable)
                    - error distribution used to extract anomaly data
                    - a mask of each contour is generated on top of error distribution individually
                    - Future plan involves using err_dist to generate contours by itself, therefore depreciating the heatMap argument
            color_threshold: 
                    - integer
                    - threshold for generating binary images (minimum darkness before being converted into a black-colored pixel)
                    - passed into cv2.threshold to convert inverted heatmap (grayscale) into binary image
            size_fail:
                    - integer (number of pixels)
                    - threshold for failing contours
                    - passed into conditional (if the contour_area is greater than the size_fail, then that anomaly is considered too big to pass)
            size_ignore:
                    - integer (number of pixels)
                    - threshold for ignoring contours
                    - passed into conditional at the start of contour loop (if contour_area is greater than size_ignore, it's too big to be an actual anomaly)
                    this assumes detected anomalies are usually a certain size
        RETURNS:
            decision:
                - a (PASS/FAIL) string used in the final decision of the ad model
            result:
                - a list of all the contours accepted before FAIL (contains every contour if PASS)
        """

        # preprocess original heatmap image
        heatmap_gray = cv2.cvtColor(heatMap.astype(np.float32), cv2.COLOR_BGR2GRAY) # turn image into grayscale
        heatmap_gray_invert = cv2.bitwise_not(heatmap_gray.astype(np.uint8)) # flip the colors
        _, heat_map_binary = cv2.threshold(heatmap_gray_invert, color_threshold, 255, 0) # turns the flipped grayscale into binary for easier fill

        heat_map_binary_fill = np.copy(heat_map_binary)

        # fill empty white spaces starting from fill_start
        fill_start = (0,0)
        floodMask = np.zeros((heat_map_binary.shape[0] + 2, heat_map_binary.shape[1] + 2), dtype=np.uint8)
        cv2.floodFill(heat_map_binary_fill, floodMask, fill_start, 0, (50, 50, 50))

        contours, _ = cv2.findContours(heat_map_binary_fill.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # finds the contours

        logger.info(f'contours length: {len(contours)}')

        result = []

        for contour in contours:
            contour_area = cv2.contourArea(contour)
            logger.info(f'contour area: {contour_area}')

            if size_ignore is not None:
                if contour_area > size_ignore:
                    continue
            
            result.append(contour)

            """
            If you wish to extract more data from the contours (mean, max, etc.), uncomment code
            and add the variables.
            
            code below is commented out to save memory

            This code below adds a mask over the contour and extracts data strictly from the selected contour region:

                mask = np.zeros_like(heatmap_for_contour).astype(np.uint8)
                cv2.drawContours(mask, [contour], 0, 255, -1)
                masked_heatmap = cv2.bitwise_and(heatmap_for_contour.astype(np.uint8), mask)
                anomaly_scores = masked_heatmap[mask > 0] 

                # mean_score = np.mean(anomaly_scores)
            """
            if contour_area > size_fail:
                return FAIL, result

        return PASS, result
    

    
    def postprocess(self, orig_image, anomaly_map, err_thresh, err_size, mask=None, useContours=False, size_fail=None, size_ignore=None, color_threshold=None, useAnnotation=True):
        """
        DESCRIPTION:
            postprocess: used to get anomaly decision and resulting image
        ARGUMENTS:
            orig_image:
                - float32 image
                - used to get original image dimensions for resizing heatmap and/or annotation method
                - height and width are extracted, then used to resize heatmap/annotated image if processContours or annotate is used 
            anomaly_map:
                - np array
                - used for masking, contouring, and/or annotation (if needed)
            err_thresh:
                - integer
                - error threshold for anomaly scores
            err_size:
                - integer (number of pixels)
                - for pass/fail condition when comparing aggregation of error scores
                - if the aggregation of the error scores combined is greater than err_size, then the decision will be 'FAIL'
            mask:
                - list of binary images
                - mask regions to ignore when determining anomaly score
                - masks of the regions are applied to the anomaly_map to be ignored
            useContours:
                - boolean
                - switch for turning on/off processContours logic
                - passed into conditional, if useContours=True, then the decision is made using processContours (through contour analysis)

                child variables:
                    size_fail: threshold for failing contours based off of size (anomaly size in pixels)
                    color_threshold: threshold for converting image to binary (used to pass into processContours, see @processContours)
                    size_ignore [OPTIONAL]: minimum size to ignore in contours (if there will be no extra large anomalous regions)
            useAnnotation:
                - boolean
                - switch for turning on/off annotation (returns original image if useAnnotation=False)
    """
        
        h,w = orig_image.shape[:2]
        anomaly_map = np.squeeze(anomaly_map.transpose((0,2,3,1)))
        if mask is not None:
            mask = cv2.resize(mask, self.bindings['input'].shape[-2:])
            anomaly_map_fp32 = anomaly_map.astype(np.float32)
            anomaly_map = cv2.bitwise_and(anomaly_map_fp32, anomaly_map_fp32, mask=mask)
        ind = anomaly_map<err_thresh
        err_count = np.count_nonzero(ind==False)
        anomaly_map[ind] = err_thresh
        max_error = {'emax':round(anomaly_map.max().tolist(), 1), 'ecnt':err_count}

        if useContours:
            if size_fail is None or color_threshold is None:
                raise ValueError("Required parameters (size_fail && contour_color_threshold) must be provided when useContours=True")
            heat_map = anomaly_map
            heat_map_rsz = cv2.resize(heat_map.astype(np.uint8), (w, h))
            residual_gray = (AnomalyModel.normalize_anomaly_map(heat_map_rsz)*255).astype(np.uint8)
            residual_bgr = cv2.applyColorMap(np.expand_dims(residual_gray,-1), cv2.COLORMAP_TURBO).astype(np.float32)
            decision, contours = self.processContours(residual_bgr, anomaly_map, color_threshold, size_fail, size_ignore)
        else:
            if err_count<=err_size:
                decision=PASS
            else:
                decision=FAIL
        
        final_image = orig_image
        if useAnnotation:
            annot = AnomalyModel.annotate(orig_image.astype(np.uint8), cv2.resize(anomaly_map.astype(np.uint8), (w, h)))
            cv2.putText(annot,
                        text=f'ad:{decision},'+ str(max_error).strip("{}").replace(" ","").replace("\'",""),
                        org=(4,h-20), fontFace=0, fontScale=1, color=[225, 255, 255],
                        thickness=2,
                        lineType=cv2.LINE_AA)
            if useContours:
                cv2.drawContours(annot, contours, -1, (255, 255, 255), 1)
            final_image = annot

        return decision, final_image, max_error

    @staticmethod
    def convert_trt(onnx_path, out_engine_path, fp16=True, workspace=4096):
        """
        convert an onnx to trt engine
        """
        logger.info(f"converting {onnx_path}...")
        assert(out_engine_path.endswith(".engine")), f"trt engine file must end with '.engine'"
        target_dir = os.path.dirname(out_engine_path)
        if not os.path.isdir(target_dir):
            os.makedirs(target_dir)
        convert_cmd = (f'trtexec --onnx={onnx_path} --saveEngine={out_engine_path}'
                       ' --explicitBatch --inputIOFormats=fp16:chw --outputIOFormats=fp16:chw'
                       f' --workspace={workspace}') + (' --fp16' if fp16 else ' ')
        os.system(convert_cmd)
        os.system(f"cp {os.path.dirname(onnx_path)}/metadata.json {os.path.dirname(out_engine_path)}")

    def from_numpy(self, x):
        return torch.from_numpy(x).to(self.device) if isinstance(x, np.ndarray) else x
    
    @staticmethod
    def normalize(image: np.ndarray) -> np.ndarray:
        """
        Normalize the image to the given mean and standard deviation for consistency with pytorch backbone
        """
        image = image.astype(np.float32)
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        image /= 255.0
        image -= mean
        image /= std
        return image
    
    @staticmethod
    def normalize_anomaly_map(data, mi=None, ma=None):
        """  normalize to [0,1) """
        data_min=mi or data.min()
        data_max=ma or data.max()
        return (data - data_min) / (data_max - data_min + 1e-16)

    @staticmethod
    def annotate(img_original, heat_map_rsz):
        residual_gray = (AnomalyModel.normalize_anomaly_map(heat_map_rsz)*255).astype(np.uint8)
        residual_bgr = cv2.applyColorMap(np.expand_dims(residual_gray,-1), cv2.COLORMAP_TURBO)
        residual_rgb = cv2.cvtColor(residual_bgr, cv2.COLOR_BGR2RGB)
        annot = cv2.addWeighted(img_original, 0.5, residual_rgb, 0.5, 0)
        ind = heat_map_rsz==0
        annot[ind] = img_original[ind]
        return annot

def test(engine_dir, images_path, annot_dir):
    """test trt engine"""

    from ad_utils import plot_fig,plot_histogram
    import time

    logger.info(f"input engine_dir is {engine_dir}")

    images = glob.glob(f"{images_path}/*.png")
    logger.info(f"{len(images)} images from {images_path}")
    if not images:
        return

    engine_path = os.path.join(engine_dir, "model.engine")
    if not os.path.isfile(engine_path):
        all = sorted([x for x in os.walk(engine_dir)][0][1])
        if all:
            engine_path = os.path.join(engine_dir, all[-1], "model.engine")
    assert(os.path.isfile(engine_path)), f"engine file does not exist - {engine_path}"

    logger.info(f"testing engine_path {engine_path}...")

    pc = AnomalyModel(engine_path)

    out_path = f"/app/out/predictions/{model}/{os.path.basename(images_path)}"
    if os.path.isdir(out_path):
        shutil.rmtree(out_path)
    os.makedirs(out_path)

    pc.warmup()

    proctime = []
    for i in range(1):
        for image_path in images:
            img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
            t0 = time.time()
            decision, annotation, outputs = pc.predict(img) # 16.5859 56.07148
            proctime.append(time.time() - t0)
            logger.info(f"decision {decision},\toutputs {outputs}")
            annotation = cv2.cvtColor(annotation, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"{out_path}/{os.path.basename(image_path)}", annotation.astype(np.uint8))
        
    if proctime:
        proctime = np.asarray(proctime)
        logger.info(f'Min Proc Time: {proctime.min()}')
        logger.info(f'Max Proc Time: {proctime.max()}')
        logger.info(f'Avg Proc Time: {proctime.mean()}')
        logger.info(f'Median Proc Time: {np.median(proctime)}')
    logger.info(f"Test results saved to {out_path}")

def convert(onnx_path, engine_dir, fp16=True):
    engine_out_path = f'{engine_dir}/model.engine'
    AnomalyModel.convert_trt(onnx_path, engine_out_path, fp16)

if __name__ == '__main__':
    import argparse
    import os

    ap = argparse.ArgumentParser()
    ap.add_argument('-a','--action', default=None, help='Action: convert, test')
    ap.add_argument('-x','--onnx_file', default=None, help='Onnx file directory.')
    ap.add_argument('-e','--engine_file', default=None, help='Engine file directory.')
    ap.add_argument('-d','--data_dir', default=None, help='Data file directory.')
    ap.add_argument('-o','--annot_dir', default=None, help='Annot file directory.')

    args = vars(ap.parse_args())
    action=args['action']
    if action=='convert':
        if not os.path.isfile(args['onnx_file']):
            raise Exception('Need a valid onnx file to generate engine.')
        if not os.path.exists(args['engine_dir']):
            os.makedirs(args['engine_dir'])
        convert(args['onnx_file'],args['engine_dir'],fp16=True)

    if action=='test':
        if not os.path.isfile(args['engine_file']):
            raise Exception('Need a valid engine file to test model.')
        if not os.path.exists(args['annot_dir']):
            os.makedirs(args['annot_dir'])
        test(args['engine_file'], args['data_dir'], args['annot_dir'])
        
    