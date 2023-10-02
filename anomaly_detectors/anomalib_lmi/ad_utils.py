import os
import logging
import time
import copy
import matplotlib
from matplotlib import pyplot as plt
from scipy.stats import kstest
import numpy as np
import cv2
from anomaly_model import AnomalyModel

logging.basicConfig(level=logging.INFO)

def plot_histogram(xvec):
    '''
    DESCRIPTION: 
        Helper function for understanding and debugging.  
        Plots histogram of xvec so that we can visualize the embedding vector distribution from each patch in the image 

    ARGS:
        xvec: single dimension numpy array for the error distance across all pixels in the input image
    '''
    # An "interface" to matplotlib.axes.Axes.hist() method
    figure=plt.figure()
    n, bins, patches = plt.hist(x=xvec, bins='auto', color='#0504aa',alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Anomaly Histogram')
    plt.text(23, 45, f'$\mu={xvec.mean()}, $\sigma={xvec.std()}')
    maxfreq = n.max()
    # Set a clean upper y-axis limit.
    plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    # plt.show()
    figure.savefig('./test_hist.png')

def test_dist(H0,sample,alpha=0.05):
    stat, p = kstest(sample, 'norm', args=(H0[0], H0[1]))
    confirm_H0=True
    if p > alpha:
        logging.info(f'Sample could belong to the known normal distribution (fail to reject H0) with p={p}')
    else:
        logging.info(f'Sample does not belong to the known normal distribution (reject H0) with p={p}')
        confirm_H0=False

    return confirm_H0


def plot_fig(predict_results, save_dir, err_thresh=None, err_max=None):
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

def processContours(self, heatMap, err_dist, color_threshold, size_fail, size_ignore=None):
    PASS = 'PASS'
    FAIL = 'FAIL'
    logger=logging.getLogger()
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
    PASS = 'PASS'
    FAIL = 'FAIL'
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
        # cv2.putText(annot,
        #             text=f'ad:{decision},'+ str(max_error).strip("{}").replace(" ","").replace("\'",""),
        #             org=(4,h-20), fontFace=0, fontScale=1, color=[225, 255, 255],
        #             thickness=2,
        #             lineType=cv2.LINE_AA)
        if useContours:
            cv2.drawContours(annot, contours, -1, (255, 255, 255), 1)
        final_image = annot

    return decision, final_image, max_error
