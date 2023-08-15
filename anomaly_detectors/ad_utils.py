import os
import logging
import time
import copy
import matplotlib
from matplotlib import pyplot as plt

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


def plot_fig(predict_results, err_mean, err_std, save_dir, err_thresh=None):
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
        fname=fname.decode('ascii')
        fname=os.path.splitext(fname)[0]
        err_dist=np.squeeze(err_dist)
        err_mean=err_dist.mean()
        err_std=err_dist.std()
        err_max=err_dist.max()
        if err_thresh is None:
            err_thresh=err_mean+3*err_std
        if err_max<=1.1*err_thresh:
            err_max=err_thresh*2
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
        ax_img[2].imshow(img.astype(int), cmap='gray', interpolation='none')
        ax=ax_img[2].imshow(heat_map, cmap='jet', alpha=0.5, interpolation='none',vmin=err_thresh,vmax=err_max)
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