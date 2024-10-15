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

class Anomalib_Base(ABC):
    
    logger = logging.getLogger('Anomalib Base')
    logger.setLevel(logging.INFO)
    
    
    @abstractmethod
    def __init__(self) -> None:
        pass
    
    
    @torch.inference_mode()
    def from_numpy(self, x):
        """
        convert numpy array to torch tensor
        """
        return torch.from_numpy(x).to(self.device) if isinstance(x, np.ndarray) else x


    def convert_to_onnx(self, export_path, opset_version=14):
        '''
        Desc: Convert existing .pt file to onnx
        Args:
            - path to output .onnx file
            - opset_version: onnx version ID
        
        '''
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
        
    
    @classmethod
    def convert_trt(cls, onnx_path, out_engine_path, fp16, workspace=4096):
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
            cmd2 = [f"cp {onnx_dir}/metadata.json {out_dir}"]
            subprocess.run(cmd2, shell=True)
        else:
            cls.logger.warning(f"metadata.json not found in {onnx_dir}")
            
            
    @classmethod
    def convert(cls, model_path, export_path, fp16):
        '''
        Desc: Converts .onnx or .pt file to tensorRT engine
        
        Args:
            - model path: model file path .pt or .onnx
            - export path: engine file path
            - fp16: floating point number length
        '''
        if os.path.isfile(export_path):
            raise Exception('Export path should be a directory.')
        ext = os.path.splitext(model_path)[1]
        if ext == '.onnx':
            cls.logger.info('Converting onnx to trt...')
            trt_path = os.path.join(export_path, 'model.engine')
            cls.convert_trt(model_path, trt_path, fp16)
        elif ext == '.pt':
            model = cls(model_path)
            # convert to onnx
            cls.logger.info('Converting pt to onnx...')
            onnx_path = os.path.join(export_path, 'model.onnx')
            model.convert_to_onnx(onnx_path)
            cls.logger.info(f'the onnx model is saved at {onnx_path}')
            # # convert to trt
            cls.logger.info('Converting onnx to trt engine...')
            trt_path = os.path.join(export_path, 'model.engine')
            cls.convert_trt(onnx_path, trt_path, fp16)
            
            
    @staticmethod
    def annotate(img, ad_scores, ad_threshold, ad_max):
        # Resize AD score to match input image
        h_img,w_img=img.shape[:2]
        ad_scores=pipeline_utils.resize_image(ad_scores,H=h_img,W=w_img)
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
    
    
    @classmethod
    def test(cls, engine_path, images_path, annot_dir,generate_stats=True,annotate_inputs=True,anom_threshold=None,anom_max=None):
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
        cls.logger.info(f"{len(images)} images from {images_path}")
        if not images:
            return
        
        # Output overhead
        out_path = annot_dir
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        # Load model from .pt or .engine
        pc = cls(engine_path)
        pc.warmup()

        proctime = []
        img_all,anom_all,fname_all,path_all=[],[],[],[]
        for image_path in images:
            cls.logger.info(f"Processing image: {image_path}.")
            image_path=str(image_path)
            img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
            t0 = time.time()
            anom_map = pc.predict(img)
            proctime.append(time.time() - t0)
            fname=os.path.split(image_path)[1]
            h,w = pc.shape_inspection
            img_preproc=pipeline_utils.resize_image(img, H=h, W=w)
            img_all.append(img_preproc)
            anom_all.append(anom_map)
            fname_all.append(fname)
            path_all.append(image_path)
        
        if generate_stats:
            # Compute & Validate pdf
            cls.logger.info(f"Computing anomaly score PDF for all data.")
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
                cls.logger.warning(f'Patch quantile saturated with max anomaly score: {max_data}, reducing to {max_data/2}')
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
            cls.logger.info('Threshold options:\n'+tp_print)

        if annotate_inputs:     
            if anom_threshold is None and generate_stats: 
                anom_threshold=gamma.ppf(0.5,alpha_hat,loc=loc_hat,scale=beta_hat)
                cls.logger.info(f'Anomaly patch threshold for 50% patch failure rate:{anom_threshold}')
            if anom_max is None and generate_stats:
                # Sample target hard coded for 3% failure rate
                p_sample_target=0.03
                if p_sample_target > quantile_sample.min():
                    # estimate p_patch from target p_sample_target
                    p_target=find_p(threshold,quantile_patch,quantile_sample, p_sample_target)
                    # find the anomaly score coresponding to that p_patch    
                    anom_max = gamma.ppf(1-p_target,alpha_hat,loc=loc_hat,scale=beta_hat)
                    cls.logger.info(f'Anomaly max set to 97 percentile:{anom_max}')
                else:
                    anom_max=threshold.max()
                    cls.logger.warning(f'Anomaly patch max set to minimum discernable value: {anom_max} due to vanishing gradient in the patch quantile.  Sample failure rate: {quantile_sample.min()*100:.2e}')
                    
            results=zip(img_all,anom_all,fname_all)
            cls.plot_fig(results,annot_dir,err_thresh=anom_threshold,err_max=anom_max)
            
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
            cls.logger.info(f'Min Proc Time: {proctime.min()}')
            cls.logger.info(f'Max Proc Time: {proctime.max()}')
            cls.logger.info(f'Avg Proc Time: {proctime.mean()}')
            cls.logger.info(f'Median Proc Time: {np.median(proctime)}')
        cls.logger.info(f"Test results saved to {out_path}")
        if generate_stats:
            # Repeat error table
            cls.logger.info('Threshold options:\n'+tp_print)
            
            
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
    