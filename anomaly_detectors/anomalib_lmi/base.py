import os
import cv2
import numpy as np
import logging
import torch
import json
import subprocess
from abc import ABC, abstractmethod

import gadget_utils.pipeline_utils as pipeline_utils


logging.basicConfig()


MINIMUM_QUANT=1e-12


def to_list(data):
    """convert to a two element list
    
    Args:
        data (int | list): a int or a two element list
        
    Returns:
        list: _description_
    """
    if isinstance(data, int):
        return [data]*2
    if len(data) != 2:
        raise Exception(f'Must be a two element list, but got {data}')
    return list(data)


class Anomalib_Base(ABC):
    
    logger = logging.getLogger('Anomalib Base')
    logger.setLevel(logging.INFO)
    
    tiler = None
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    @abstractmethod
    def __init__(self) -> None:
        pass
    
    
    @torch.inference_mode()
    def from_numpy(self, x):
        """
        convert numpy array to torch tensor
        """
        return torch.from_numpy(x).to(self.device) if isinstance(x, np.ndarray) else x


    def convert_to_onnx(self, export_path, input_hw=None, opset_version=14):
        '''
        Desc: Convert existing .pt file to onnx
        Args:
            - path to output .onnx file
            - input_hw: a int if h==w or a list of (h,w)
            - opset_version: onnx version ID
        '''
        # write metadata to export path
        with open(os.path.join(os.path.dirname(export_path), "metadata.json"), "w", encoding="utf-8") as metadata_file:
            json.dump(self.pt_metadata, metadata_file, ensure_ascii=False, indent=4)
        
        if self.tiler is not None:
            if input_hw is None:
                raise Exception('Must provide input (h,w) when convert model to onnx and use tiling')
            hw = to_list(input_hw)
            zeros = torch.zeros(1,3,*hw,device=self.device)
            tiles = self.tiler.tile(zeros)
            b,c,h,w = tiles.shape
        else:
            b,c = 1,3
            h,w = self.model_shape
        torch.onnx.export(
            self.pt_model,
            torch.zeros((b, c, h, w)).to(self.device),
            export_path,
            opset_version=opset_version,
            input_names=["input"],
            output_names=["output"],
        )
        
    
    def convert_trt(self, onnx_path, out_engine_path, fp16, workspace=4096):
        """
        Desc: Convert an onnx to trt engine
        Args:
            - onnx_path: input file path
            - out_engine_path: output file path
            - fp16: set fixed point width
            - workspace: conversion memory size in MB
        """
        if not out_engine_path.endswith(".engine"):
            raise Exception(f"trt engine file must end with '.engine'")
        
        out_dir = os.path.dirname(out_engine_path)
        os.makedirs(out_dir, exist_ok=True)
        
        # run convert cmd
        cmd = [
            'trtexec',
            f'--onnx={onnx_path}',
            f'--saveEngine={out_engine_path}',
            f'--memPoolSize=workspace:{workspace}'
        ]
        if fp16:
            cmd.append('--fp16')
        subprocess.run(cmd, check=True)
        
        # check if metadata.json exists in the same directory as onnx_path
        onnx_dir = os.path.dirname(onnx_path)
        if os.path.isfile(f"{onnx_dir}/metadata.json"):
            cmd2 = [f"cp -sf {onnx_dir}/metadata.json {out_dir}"]
            subprocess.run(cmd2, shell=True)
        else:
            self.logger.warning(f"metadata.json not found in {onnx_dir}")
            
            
    def convert(self, model_path, export_path, input_hw=None, fp16=True):
        '''
        Desc: Converts .onnx or .pt file to tensorRT engine
        
        Args:
            - model path: model file path .pt or .onnx
            - export path: engine file path
            - input_hw: a int if h==w or a list of input height and width
            - fp16: floating point number length
        '''
        if os.path.isfile(export_path):
            raise Exception('Export path should be a directory.')
        ext = os.path.splitext(model_path)[1]
        if ext == '.onnx':
            self.logger.info('Converting onnx to trt...')
            trt_path = os.path.join(export_path, 'model.engine')
            self.convert_trt(model_path, trt_path, fp16)
        elif ext == '.pt':
            # convert to onnx
            self.logger.info('Converting pt to onnx...')
            onnx_path = os.path.join(export_path, 'model.onnx')
            self.convert_to_onnx(onnx_path, input_hw)
            self.logger.info(f'the onnx model is saved at {onnx_path}')
            # # convert to trt
            self.logger.info('Converting onnx to trt engine...')
            trt_path = os.path.join(export_path, 'model.engine')
            self.convert_trt(onnx_path, trt_path, fp16)
            
            
    @torch.inference_mode()
    def annotate(self, img, ad_scores, ad_threshold, ad_max):
        """generate an annotated image 

        Args:
            img (numpy | tensor): an intensity image with a shape of [h,w,c]
            ad_scores (numpy | tensor): an error distance map with a shape of [h,w]
            ad_threshold (float): threshold for determining anomaly area
            ad_max (float): max AD score for normalizing the annotated images
            
        Returns:
            numpy: an annotated image
        """
        # convert to tensor
        ad_scores = self.from_numpy(ad_scores)
        img = self.from_numpy(img)
        ad_threshold = self.from_numpy(np.array(ad_threshold)).to(ad_scores.dtype)
        ad_max = self.from_numpy(np.array(ad_max)).to(ad_scores.dtype)
        # Resize AD score to match input image
        h_img,w_img=img.shape[:2]
        ad_scores=pipeline_utils.resize_image(ad_scores,H=h_img,W=w_img)
        # shrink the min-max range
        ad_scores[ad_scores<ad_threshold] = ad_threshold
        ad_scores[ad_scores>ad_max]=ad_max
        # apply colormap
        ad_norm=(ad_scores-ad_threshold)/(ad_max-ad_threshold)
        ad_gray=(ad_norm*255).to(torch.uint8)
        ad_bgr = cv2.applyColorMap(np.expand_dims(ad_gray.cpu().numpy(),-1), cv2.COLORMAP_TURBO)
        residual_rgb = cv2.cvtColor(ad_bgr, cv2.COLOR_BGR2RGB)
        residual_rgb = self.from_numpy(residual_rgb)
        
        # Overlay anomaly heat map with input image
        annot = img*0.6 + residual_rgb*0.4
        annot = annot.round().to(torch.uint8)
        img = img.to(torch.uint8)
        m = ad_gray==0
        # replace all below-threshold pixels with input image indicating no anomaly
        annot[m] = img[m]
        return annot.cpu().numpy()
    
    
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
    
    
    def test(self, images_path, annot_dir,generate_stats=True,annotate_inputs=True,anom_threshold=None,anom_max=None):
        """
        Desc: test model performance
        Args:
            - engine_path: .pt or .engine file path
            - images_path: Path to image data
            - annot_dir: Path to annotation data dir
            - generate_stats: Fit gamma distribution to all data in dataset.  Propose resonable thresholds for different failure rates. 
            - annotate_inputs: option to show anomaly score histogram and heat map for each image in thd dataset (def: True)
            - anom_threshold: user defined anomaly threshold (sets beginning of heat map)
            - anom_max: user defined anomaly max (sets end of the heat map)
        """
        from pathlib import Path
        import time
        from scipy.stats import gamma
        from scipy import interpolate
        import matplotlib.pyplot as plt
        from tabulate import tabulate
        import csv
        from anomalib_lmi.ad_utils import plot_fig
        
        def find_p(thresh_array,p_patch_array,p_sample_array, p_sample_target):
            '''
            Desc: Find the p-value that acheives the desired sample failure rate.  We start by estimating the threshold from the empiracal p_sample_array.  Then we use that threshold to estimate the corresponding p_patch.
            
            Args: 
                - thresh_array: input threshold array 
                - p_patch_array: corresponding p-value at the patch level (generated using gamma dist model)
                - p_sample_array: corresponding p-value at the sample level (generated empirically)
                - p_sample_target: desired sample level p-value 
            '''
            x1=p_sample_array
            x2=thresh_array
            x3=p_patch_array
            # interpolation function to find threshold for a specified p_sample
            f1=interpolate.interp1d(x1,x2)
            # estimate the threshold for p_sample_target 
            thresh_target=f1(p_sample_target)
            # interpolation function to find p_patch for a specified threshold
            f2=interpolate.interp1d(x2,x3)
            # find the threshold for the p_patch that corresponds to p_sample_target
            p_target=f2(thresh_target)
            return p_target

        # Input data
        directory_path=Path(images_path)
        images=list(directory_path.rglob('*.png')) + list(directory_path.rglob('*.jpg'))
        self.logger.info(f"{len(images)} images from {images_path}")
        if not images:
            return
        
        # Output overhead
        out_path = annot_dir
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        proctime = []
        img_all,anom_all,fname_all,path_all=[],[],[],[]
        for image_path in images:
            self.logger.info(f"Processing image: {image_path}.")
            image_path=str(image_path)
            img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
            t0 = time.time()
            anom_map = self.predict(img)
            proctime.append(time.time() - t0)
            fname=os.path.split(image_path)[1]
            if self.tiler is None:
                h,w = self.model_shape
                img=pipeline_utils.resize_image(img, H=h, W=w)
            img_all.append(img)
            anom_all.append(anom_map)
            fname_all.append(fname)
            path_all.append(image_path)
        
        if generate_stats:
            # Compute & Validate pdf
            self.logger.info(f"Computing anomaly score PDF for all data.")
            anom_sq=np.squeeze(np.array(anom_all))
            data=np.ravel(anom_sq)
            # Fit gamma distribution to anomaly data across entire data set
            alpha_hat, loc_hat, beta_hat = gamma.fit(data, floc=0)
            # Plot histogram and gamma dist fit
            x = np.linspace(min(data), max(data), 1000)
            pdf_fitted = gamma.pdf(x, alpha_hat, loc=loc_hat, scale=beta_hat)
            plt.hist(data, bins=100, density=True, alpha=0.7, label='Observed Data')
            plt.plot(x, pdf_fitted, 'r-', label=f'Fitted Gamma')
            plt.legend()
            plt.savefig(os.path.join(annot_dir,'gamma_pdf_fit.png'))
            max_data=max(data)
            # Generate uniform anomaly threshold samples across available anomaly score range
            threshold = np.linspace(min(data), max_data, 10)
            # Determine percentage of failed parts for each threshold
            quantile_patch = 1 - gamma.cdf(threshold, alpha_hat, loc=loc_hat, scale=beta_hat)
            # Reduce threshold range when threshold values are too far into the tail of the gamma distribution (quantile goes to zero)
            while quantile_patch.min()<MINIMUM_QUANT:
                self.logger.warning(f'Patch quantile saturated with max anomaly score: {max_data}, reducing to {max_data/2}')
                max_data=max_data/1.2
                threshold = np.linspace(min(data), max_data, 10)
                quantile_patch = 1 - gamma.cdf(threshold, alpha_hat, loc=loc_hat, scale=beta_hat)
            # Extract patch level distribution table data
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
            # Print statistics
            tp_print=tabulate(tp, tablefmt='grid')
            self.logger.info('Threshold options:\n'+tp_print)

        if annotate_inputs:     
            if anom_threshold is None and generate_stats: 
                anom_threshold=gamma.ppf(0.5,alpha_hat,loc=loc_hat,scale=beta_hat)
                self.logger.info(f'Anomaly patch threshold for 50% patch failure rate:{anom_threshold}')
            if anom_max is None and generate_stats:
                # Sample target hard coded for 3% failure rate
                p_sample_target=0.03
                if p_sample_target > quantile_sample.min():
                    # estimate p_patch from target p_sample_target
                    p_target=find_p(threshold,quantile_patch,quantile_sample, p_sample_target)
                    # find the anomaly score coresponding to that p_patch    
                    anom_max = gamma.ppf(1-p_target,alpha_hat,loc=loc_hat,scale=beta_hat)
                    self.logger.info(f'Anomaly max set to 97 percentile:{anom_max}')
                else:
                    anom_max=threshold.max()
                    self.logger.warning(f'Anomaly patch max set to minimum discernable value: {anom_max} due to vanishing gradient in the patch quantile.  Sample failure rate: {quantile_sample.min()*100:.2e}')
                    
            results=zip(img_all,anom_all,fname_all)
            plot_fig(results,annot_dir,err_thresh=anom_threshold,err_max=anom_max)
            
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
            self.logger.info(f'Min Proc Time: {proctime.min()}')
            self.logger.info(f'Max Proc Time: {proctime.max()}')
            self.logger.info(f'Avg Proc Time: {proctime.mean()}')
            self.logger.info(f'Median Proc Time: {np.median(proctime)}')
        self.logger.info(f"Test results saved to {out_path}")
        if generate_stats:
            # Repeat error table
            self.logger.info('Threshold options:\n'+tp_print)
            
    