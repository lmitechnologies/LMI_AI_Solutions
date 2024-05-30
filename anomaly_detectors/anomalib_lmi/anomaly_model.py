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
import albumentations as A
import json
from image_utils.img_resize import resize


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('AnomalyModel')

PASS = 'PASS'
FAIL = 'FAIL'
MINIMUM_QUANT=1e-12

Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))

class AnomalyModel:
    '''
    Desc: Class used for AD model inference.  
     
    Args: 
        - model_path: path to .pt file or TRT engine
    
    '''
    
    def __init__(self, model_path):
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            logger.warning('GPU device unavailable. Use CPU instead.')
            self.device = torch.device('cpu')
        _,ext=os.path.splitext(model_path)
        if ext=='.engine':
            with open(model_path, "rb") as f, trt.Runtime(trt.Logger(trt.Logger.WARNING)) as runtime:
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
            self.shape_inspection=list(shape[-2:])
            self.inference_mode='TRT'
        elif ext=='.pt':     
            model = torch.load(model_path,map_location=self.device)["model"]
            model.eval()
            self.pt_model=model.to(self.device)
            self.pt_metadata = torch.load(model_path, map_location=self.device)["metadata"] if model_path else {}
            self.pt_transform=A.from_dict(self.pt_metadata["transform"])
            for d in self.pt_metadata['transform']['transform']['transforms']:
                if d['__class_fullname__']=='Resize':
                    self.shape_inspection = [d['height'], d['width']]
            self.inference_mode='PT'
                    
    def preprocess(self, image):
        '''
        Desc: Preprocess input image.
        args:
            - image: numpy array [H,W,Ch]
        
        '''
        if self.inference_mode=='TRT':
            h, w =  self.shape_inspection
            img = cv2.resize(self.normalize(image), (w,h), interpolation=cv2.INTER_AREA)
            input_dtype = np.float16 if self.fp16 else np.float32
            input_batch = np.array(np.repeat(np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0), 1, axis=0), dtype=input_dtype)
            return self.from_numpy(input_batch)
        elif self.inference_mode=='PT':
            processed_image = self.pt_transform(image=image)["image"]
            if len(processed_image) == 3:
                processed_image = processed_image.unsqueeze(0)
            return processed_image.to(self.device)
        else:
            raise Exception(f'Unknown model format: {self.inference_mode}')

    def warmup(self):
        '''
        Desc: Warm up model using a np zeros array with shape matching model input size.
        Args: None
        '''
        logger.info("warmup started")
        t0 = time.time()
        shape=self.shape_inspection+[3,]
        self.predict(np.zeros(shape))
        logger.info(f"warmup ended - {time.time()-t0:.4f}")

    def predict(self, image):
        '''
        Desc: Model prediction 
        Args: image: numpy array [H,W,Ch]
        
        Note: predict calls the preprocess method
        '''
        if self.inference_mode=='TRT':
            input_batch = self.preprocess(image)
            self.binding_addrs['input'] = int(input_batch.data_ptr())
            self.context.execute_v2(list(self.binding_addrs.values()))
            outputs = {x:self.bindings[x].data.cpu().numpy() for x in self.output_names}
            output=outputs['output']
        elif self.inference_mode=='PT':
            preprocessed_image = self.preprocess(image)
            output=self.pt_model(preprocessed_image)[0].cpu().numpy()
        output=np.squeeze(output).astype(np.float32)
        return output


    # DEPRECATE (TO SPECIFIC)    
    # def postprocess(self,orig_image, anomaly_map, err_thresh, err_size, mask=None,info_on_annot=True):
    #     h,w = orig_image.shape[:2]
    #     anomaly_map = np.squeeze(anomaly_map.transpose((0,2,3,1)))
    #     ind = anomaly_map<err_thresh
    #     if mask is not None:
    #         h_i,w_i = self.shape_inspection
    #         mask = cv2.resize(mask, (w_i,h_i)).astype(np.uint8)
    #         # set anamaly score to 0 based on the mask
    #         np.putmask(anomaly_map, mask==0, 0)
    #     ind = anomaly_map<err_thresh
    #     err_count = np.count_nonzero(ind==False)
    #     details = {'emax':round(anomaly_map.max().tolist(), 2), 'ecnt':err_count}
    #     if err_count<=err_size:
    #         decision=PASS
    #         annot=orig_image
    #     else:
    #         decision=FAIL
    #         anomaly_map[ind] = 0
    #         annot = AnomalyModel.annotate(orig_image.astype(np.uint8), 
    #                                       cv2.resize(anomaly_map.astype(np.float32), (w, h)))
    #     if info_on_annot:
    #         cv2.putText(annot,
    #             text=f'ad:{decision},'+ str(details).strip("{}").replace(" ","").replace("\'",""),
    #             org=(4,h-20), fontFace=0, fontScale=1, color=[225, 255, 255],
    #             thickness=2,
    #             lineType=cv2.LINE_AA)
        
    #     return decision, annot, details
    
    
    def convert_to_onnx(self, export_path, opset_version=11):
        '''
        Desc: Convert existing .pt file to onnx
        Args:
            - path to output .onnx file
            - opset_version: onnx version ID
        
        '''
        if self.inference_mode!='PT':
            raise Exception('Not a pytorch model. Cannot convert to onnx.')
        
        # write metadata to export path
        with open(os.path.join(os.path.dirname(export_path), "metadata.json"), "w", encoding="utf-8") as metadata_file:
            json.dump(self.pt_metadata, metadata_file, ensure_ascii=False, indent=4)
        
        h,w = self.shape_inspection
        torch.onnx.export(
            self.pt_model,
            torch.zeros((1, 3, h, w)).to(self.device),
            export_path,
            opset_version=opset_version,
            input_names=["input"],
            output_names=["output"],
        )
        
    @staticmethod
    def annotate(img, ad_scores, ad_threshold, ad_max):
        # Resize AD score to match input image
        h_img,w_img=img.shape[:2]
        ad_scores=resize(ad_scores,height=h_img,width=w_img,inter=cv2.INTER_NEAREST)
        # Set all low score pixels to threshold to improve heat map precision
        indices=np.where(ad_scores<ad_threshold)
        ad_scores[indices]=ad_threshold
        # Set upper limit on anomaly score.
        ad_scores[ad_scores>ad_max]=ad_max
        # Generate heat map
        ad_norm=(ad_scores-ad_threshold)/(ad_max-ad_threshold)
        ad_gray=(ad_norm*255).astype(np.uint8)
        ad_bgr = cv2.applyColorMap(np.expand_dims(ad_gray,-1), cv2.COLORMAP_TURBO)
        residual_rgb = cv2.cvtColor(ad_bgr, cv2.COLOR_BGR2RGB)
        # Overlay anomaly heat map with input image
        annot = cv2.addWeighted(img.astype(np.uint8), 0.6, residual_rgb, 0.4, 0)
        indices=np.where(ad_gray==0)
        # replace all below-threshold pixels with input image indicating no anomaly
        annot[indices]=img[indices]
        return annot
    
    @staticmethod
    def compute_ad_contour_bbox(ad_scores,ad_max):
        canvas=np.zeros(ad_scores.shape,dtype=np.uint8)
        canvas[ad_scores>=ad_max]=255
        contours, _ = cv2.findContours(canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
        bboxes=[]
        for contour in sorted_contours:
            x, y, w, h = cv2.boundingRect(contour)
            bboxes.append([x,y,x+w,y+h])
        return sorted_contours, bboxes
       
    
    @staticmethod
    def convert_trt(onnx_path, out_engine_path, fp16=True, workspace=4096):
        """
        Desc: Convert an onnx to trt engine
        Args:
            - onnx_path: input file path
            - out_engine_path: output file path
            - fp16: set fixed point width (True by default)
            - workspace: conversion memory size.  (Uncertain how sensitive model/generation is to this config)
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
    
    def normalize(self,image: np.ndarray) -> np.ndarray:
        """
        Desc: Normalize the image to the given mean and standard deviation for consistency with pytorch backbone
        """
        image = image.astype(np.float32)
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        image /= 255.0
        image -= mean
        image /= std
        return image
    
    @staticmethod
    def test(engine_path, images_path, annot_dir,generate_stats=True,annotate_inputs=True,anom_threshold=None,anom_max=None):
        """
        Desc: test model performance
        Args:
            - engine_path: .pt or .engine file path
            - images_path: Path to image data
            - annot_dir: Path to annotation data dir
            - generate_stats: Fit gamma distribution to all data in dataset.  Propose resonable thresholds for different failure rates. 
            - annotate_inputs: option to show anomaly score histogram and heat map for each image in thd dataset (def: True)
            - anom_threshold: 
        """

        from pathlib import Path
        import time
        from scipy.stats import gamma
        from scipy import interpolate
        import matplotlib.pyplot as plt
        from tabulate import tabulate
        import csv

        # images = glob.glob(f"{images_path}/*.png")
        directory_path=Path(images_path)
        images=list(directory_path.rglob('*.png')) + list(directory_path.rglob('*.jpg'))
        logger.info(f"{len(images)} images from {images_path}")
        if not images:
            return
        
        logger.info(f"Loading engine: {engine_path}.")

        pc = AnomalyModel(engine_path)

        out_path = annot_dir
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        pc.warmup()

        proctime = []

        def find_p(thresh_array,p_patch_array,p_sample_array, p_sample_target):
            
            x1=p_sample_array
            x2=thresh_array
            x3=p_patch_array

            f1=interpolate.interp1d(x1,x2)
            thresh_target=f1(p_sample_target)

            f2=interpolate.interp1d(x2,x3)
            p_target=f2(thresh_target)

            return p_target

        img_all,anom_all,fname_all,path_all=[],[],[],[]
        for image_path in images:
            logger.info(f"Processing image: {image_path}.")
            image_path=str(image_path)
            img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
            t0 = time.time()
            anom_map = pc.predict(img).astype(np.float32)
            proctime.append(time.time() - t0)
            fname=os.path.split(image_path)[1]
            h,w = pc.shape_inspection
            img_preproc=cv2.resize(img, (w,h), interpolation=cv2.INTER_AREA)
            img_all.append(img_preproc)
            anom_all.append(anom_map)
            fname_all.append(fname)
            path_all.append(image_path)
        
        if generate_stats:
            # Compute & Validate pdf
            logger.info(f"Computing anomaly score PDF for all data.")
            anom_sq=np.squeeze(np.array(anom_all))
            data=np.ravel(anom_sq)
            alpha_hat, loc_hat, beta_hat = gamma.fit(data, floc=0)
            x = np.linspace(min(data), max(data), 1000)
            pdf_fitted = gamma.pdf(x, alpha_hat, loc=loc_hat, scale=beta_hat)
            plt.hist(data, bins=100, density=True, alpha=0.7, label='Observed Data')
            plt.plot(x, pdf_fitted, 'r-', label=f'Fitted Gamma')
            plt.legend()
            plt.savefig(os.path.join(annot_dir,'gamma_pdf_fit.png'))
            max_data=max(data)
            threshold = np.linspace(min(data), max_data, 10)
            quantile_patch = 1 - gamma.cdf(threshold, alpha_hat, loc=loc_hat, scale=beta_hat)
            while quantile_patch.min()<MINIMUM_QUANT:
                logger.warning(f'Patch quantile saturated with max anomaly score: {max_data}, reducing to {max_data/2}')
                max_data=max_data/1.2
                threshold = np.linspace(min(data), max_data, 10)
                quantile_patch = 1 - gamma.cdf(threshold, alpha_hat, loc=loc_hat, scale=beta_hat)
            quantile_patch_str=["{:.{}e}".format(item*100, 2) for item in np.squeeze(quantile_patch).tolist()]
            quantile_patch_str=['Prob of Patch Defect']+quantile_patch_str
            quantile_sample_str=['Prob of Sample Defect']
            quantile_sample=[]
            for t in threshold:
                ind=np.where(anom_sq>t)
                ind_u=np.unique(ind[0])
                percent=len(ind_u)/len(fname_all)
                quantile_sample.append(percent)
                quantile_sample_str.append("{:.{}e}".format(percent*100, 2))

            quantile_sample=np.array(quantile_sample)
            threshold_str=["{:.{}e}".format(item, 2) for item in np.squeeze(threshold).tolist()]
            threshold_str=['Threshold']+threshold_str    
            
            tp=[threshold_str,quantile_patch_str,quantile_sample_str]
            

            tp_print=tabulate(tp, tablefmt='grid')
            logger.info('Threshold options:\n'+tp_print)

        if annotate_inputs:
            
            if anom_threshold is None and generate_stats: 
                anom_threshold=gamma.ppf(0.5,alpha_hat,loc=loc_hat,scale=beta_hat)
                logger.info(f'Anomaly patch threshold for 50% patch failure rate:{anom_threshold}')
            if anom_max is None and generate_stats:
                p_sample_target=0.03
                if p_sample_target > quantile_sample.min():
                    p_target=find_p(threshold,quantile_patch,quantile_sample, p_sample_target)    
                    anom_max = gamma.ppf(1-p_target,alpha_hat,loc=loc_hat,scale=beta_hat)
                    logger.info(f'Anomaly  max set to 95 percentile:{anom_max}')
                else:
                    anom_max=threshold.max()
                    logger.warning(f'Anomaly patch max set to minimum discernable value: {anom_max} due to vanishing gradient in the patch quantile.  Sample failure rate: {quantile_sample.min()*100:.2e}')
                    
            results=zip(img_all,anom_all,fname_all)
            AnomalyModel.plot_fig(results,annot_dir,err_thresh=anom_threshold,err_max=anom_max)
            
        # get anom stats
        means = np.array([anom.mean() for anom in anom_all])
        maxs = np.array([anom.max() for anom in anom_all])
        stds = np.array([np.std(anom) for anom in anom_all])
        
        # sort based on anom maxs
        idx = np.argsort(maxs)[::-1]
        maxs = maxs[idx]
        means = means[idx]
        stds = stds[idx]
        fname_all = np.array(fname_all)[idx]
        
        # write to a csv file
        with open(os.path.join(annot_dir,'stats.csv'), 'w') as csvfile:
            fieldnames = ['fname', 'mean', 'max', 'std']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for data in zip(fname_all,means,maxs,stds):
                tmp_dict = {f:d for f,d in zip(fieldnames,data)}
                writer.writerow(tmp_dict)
            
        if proctime:
            proctime = np.asarray(proctime)
            logger.info(f'Min Proc Time: {proctime.min()}')
            logger.info(f'Max Proc Time: {proctime.max()}')
            logger.info(f'Avg Proc Time: {proctime.mean()}')
            logger.info(f'Median Proc Time: {np.median(proctime)}')
        logger.info(f"Test results saved to {out_path}")
        if generate_stats:
            # Repeat error table
            logger.info('Threshold options:\n'+tp_print)

    @staticmethod
    def convert(model_path, export_path, fp16=True):
        '''
        Desc: Converts .onnx or .pt file to tensorRT engine
        
        Args:
            - model path: model file path .pt or .onnx
            - export path: engine file path
            - fp16: fixed point number length
        '''
        if os.path.isfile(export_path):
            raise Exception('Export path should be a directory.')
        ext = os.path.splitext(model_path)[1]
        if ext == '.onnx':
            logger.info('Converting onnx to trt...')
            trt_path = os.path.join(export_path, 'model.engine')
            AnomalyModel.convert_trt(model_path, trt_path, fp16)
        elif ext == '.pt':
            model = AnomalyModel(model_path)
            # convert to onnx
            logger.info('Converting pt to onnx...')
            onnx_path = os.path.join(export_path, 'model.onnx')
            model.convert_to_onnx(onnx_path)
            logger.info(f'the onnx model is saved at {onnx_path}')
            # # convert to trt
            logger.info('Converting onnx to trt engine...')
            trt_path = os.path.join(export_path, 'model.engine')
            AnomalyModel.convert_trt(onnx_path, trt_path, fp16)
            
    @staticmethod
    def plot_fig(predict_results, save_dir, err_thresh=None, err_max=None):
        from matplotlib import pyplot as plt
        '''
        DESCRIPTION: generate matplotlib figures for inspection results

        ARGS: 
            predict_results: zip object
                image_array: numpy array (batch,dim,dim,3) for all images in dataset
                error_dist_array: numpy array (batch,dim,dim,3) for normalized distances/errors
                fname_array: numpy array (batch) for desriptive filenames for each img/score
            err_mean: mean training error (~0)
            err_std: std training error
            save_dir: path to save directory
            err_ceil_z: z score for heat map normalization
    '''

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        # Assume normalized error distance
        ERR_FLOOR = 0

        for img,err_dist,fname in predict_results:
            # fname=fname.decode('ascii')
            fname,fext=os.path.splitext(fname)
            logging.info(f'Processing data for: {fname}')
            err_dist=np.squeeze(err_dist)
            err_mean=err_dist.mean()
            err_std=err_dist.std()
            if err_thresh is None:
                err_thresh=err_dist.mean()
            if err_max is None:
                err_max=err_dist.max()

            heat_map=err_dist.copy()
            heat_map[heat_map<err_thresh]=err_thresh
            fig_img, ax_img = plt.subplots(1, 3, figsize=(12, 3))
            fig_img.subplots_adjust(right=0.9)
            for ax_i in ax_img:
                ax_i.axes.xaxis.set_visible(False)
                ax_i.axes.yaxis.set_visible(False)
            ax_img[0].imshow(img.astype(int))
            ax_img[0].title.set_text('Image')
            n, bins, patches = ax_img[1].hist(x=err_dist.flatten(), bins='auto', color='#0504aa',alpha=0.7, rwidth=0.85)
            ax_img[1].axes.xaxis.set_visible(True)
            ax_img[1].axes.yaxis.set_visible(True)
            ax_img[1].grid(axis='y', alpha=0.75)
            ax_img[1].xaxis.axis_name='Error'
            ax_img[1].yaxis.axis_name='Frequency'
            ax_img[1].title.set_text('Anomaly Histogram')
            ax_img[1].text(bins.mean(), n.mean(), f'\u03BC={err_mean:0.1f}, \u03C3={err_std:0.1f}')
            ax_img[2].imshow(cv2.cvtColor(img,cv2.COLOR_RGB2GRAY), cmap='gray', interpolation='none')
            ax=ax_img[2].imshow(heat_map, cmap='jet', alpha=0.4, interpolation='none',vmin=err_thresh,vmax=err_max)
            ax_img[2].title.set_text('Anomaly Heat Map')
            # ax_img[2].imshow(mask.astype(int), cmap='gray')
            # ax_img[2].title.set_text('Predicted Mask')
            left = 0.92
            bottom = 0.15
            width = 0.015
            height = 1 - 2 * bottom
            rect = [left, bottom, width, height]
            cbar_ax = fig_img.add_axes(rect)
            cb = plt.colorbar(ax, shrink=0.6, cax=cbar_ax, fraction=0.046)
            cb.ax.tick_params(labelsize=8)
            font = {
                'family': 'serif',
                'color': 'black',
                'weight': 'normal',
                'size': 8,
            }
            cb.set_label('Anomaly Score', fontdict=font)
            filepath=os.path.join(save_dir,f'{fname}_annot.png')
            folder=os.path.split(filepath)[0]
            if not os.path.exists(folder):
                os.makedirs(folder)
            fig_img.savefig(filepath, dpi=100)
            plt.close()


if __name__ == '__main__':
    import argparse
    import os

    ap = argparse.ArgumentParser()
    ap.add_argument('-a','--action', default="test", help='Action: convert, test')
    ap.add_argument('-i','--model_path', default="/app/model/model.pt", help='Input model file path.')
    ap.add_argument('-e','--export_dir', default="/app/export")
    ap.add_argument('-d','--data_dir', default="/app/data", help='Data file directory.')
    ap.add_argument('-o','--annot_dir', default="/app/annotation_results", help='Annot file directory.')
    ap.add_argument('-g','--generate_stats', action='store_true',help='generate the data stats')
    ap.add_argument('-p','--plot',action='store_true', help='plot the annotated images')
    ap.add_argument('-t','--ad_threshold',type=float,default=None,help='AD patch threshold.')
    ap.add_argument('-m','--ad_max',type=float,default=None,help='AD patch max anomaly.')

    args = vars(ap.parse_args())
    action=args['action']
    model_path = args['model_path']
    export_dir = args['export_dir']
    if action=='convert':
        if not os.path.isfile(model_path):
            raise Exception('Cannot find the model file. Need a valid model file to convert.')
        if not os.path.exists(export_dir):
            os.makedirs(export_dir)
        AnomalyModel.convert(model_path,export_dir,fp16=True)

    if action=='test':
        if not os.path.isfile(model_path):
            raise Exception(f'Error finding {model_path}. Need a valid model file to test model.')
        if not os.path.exists(args['annot_dir']):
            os.makedirs(args['annot_dir'])
        AnomalyModel.test(model_path, args['data_dir'],
             args['annot_dir'],
             generate_stats=args['generate_stats'],
             annotate_inputs=args['plot'],
             anom_threshold=args['ad_threshold'],
             anom_max=args['ad_max'])
