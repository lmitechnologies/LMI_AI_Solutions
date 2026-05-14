import csv
import logging
import os
import time
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import interpolate
from scipy.stats import gamma
from tabulate import tabulate

import lmi_utils.gadget_utils.pipeline_utils as pipeline_utils
from lmi_utils.preprocess_utils.preprocessor import Preprocessor
from lmi_utils.preprocess_utils.reconstructor import Reconstructor

MINIMUM_QUANT = 1e-12

logger = logging.getLogger("anomalib_lmi.evaluate")


def _plot_fig(predict_results, save_dir, err_thresh=None, err_max=None):
    """Generate per-image inspection figures (image / histogram / heatmap).

    Args:
        predict_results: iterable of (image, error_distribution, filename) tuples.
        save_dir: directory to write annotated PNGs into.
        err_thresh: heatmap floor; defaults to per-image mean.
        err_max: heatmap ceiling; defaults to per-image max.
    """
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    for img, err_dist, fname in predict_results:
        fname, _fext = os.path.splitext(fname)
        err_dist = np.squeeze(err_dist)
        err_mean = err_dist.mean()
        err_std = err_dist.std()
        if err_thresh is None:
            err_thresh = err_dist.mean()
        if err_max is None:
            err_max = err_dist.max()

        heat_map = err_dist.copy()
        heat_map[heat_map < err_thresh] = err_thresh
        fig_img, ax_img = plt.subplots(1, 3, figsize=(12, 3))
        fig_img.subplots_adjust(right=0.9)
        for ax_i in ax_img:
            ax_i.axes.xaxis.set_visible(False)
            ax_i.axes.yaxis.set_visible(False)
        ax_img[0].imshow(img.astype(int))
        ax_img[0].title.set_text("Image")
        n, bins, _patches = ax_img[1].hist(x=err_dist.flatten(), bins="auto", color="#0504aa", alpha=0.7, rwidth=0.85)
        ax_img[1].axes.xaxis.set_visible(True)
        ax_img[1].axes.yaxis.set_visible(True)
        ax_img[1].grid(axis="y", alpha=0.75)
        ax_img[1].xaxis.axis_name = "Error"
        ax_img[1].yaxis.axis_name = "Frequency"
        ax_img[1].title.set_text("Anomaly Histogram")
        ax_img[1].text(bins.mean(), n.mean(), f"μ={err_mean:0.1f}, σ={err_std:0.1f}")
        ax_img[2].imshow(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), cmap="gray", interpolation="none")
        ax = ax_img[2].imshow(
            heat_map,
            cmap="jet",
            alpha=0.4,
            interpolation="none",
            vmin=err_thresh,
            vmax=err_max,
        )
        ax_img[2].title.set_text("Anomaly Heat Map")
        left = 0.92
        bottom = 0.15
        width = 0.015
        height = 1 - 2 * bottom
        rect = [left, bottom, width, height]
        cbar_ax = fig_img.add_axes(rect)
        cb = plt.colorbar(ax, shrink=0.6, cax=cbar_ax, fraction=0.046)
        cb.ax.tick_params(labelsize=8)
        font = {"family": "serif", "color": "black", "weight": "normal", "size": 8}
        cb.set_label("Anomaly Score", fontdict=font)
        filepath = os.path.join(save_dir, f"{fname}_annot.png")
        folder = os.path.split(filepath)[0]
        if not os.path.exists(folder):
            os.makedirs(folder)
        fig_img.savefig(filepath, dpi=100)
        plt.close()


def _find_p(thresh_array, p_patch_array, p_sample_array, p_sample_target):
    """Find the p_patch corresponding to a target sample-level p-value via interpolation."""
    f1 = interpolate.interp1d(p_sample_array, thresh_array)
    thresh_target = f1(p_sample_target)
    f2 = interpolate.interp1d(thresh_array, p_patch_array)
    return f2(thresh_target)


def evaluate(
    model,
    images_path,
    annot_dir,
    generate_stats=True,
    annotate_inputs=True,
    anom_threshold=None,
    anom_max=None,
    tile=None,
    stride=None,
    overlap_mode="gaussian",
    scale_mode="padding",
    limit=None,
):
    """Run inference over a directory of images and emit eval artifacts.

    Args:
        model: AD model exposing predict() and image_size.
        images_path: Path to image data.
        annot_dir: Path to annotation output directory.
        generate_stats: Fit gamma distribution and propose thresholds.
        annotate_inputs: Save heatmap/histogram plots per image.
        anom_threshold: User-defined anomaly threshold (heatmap start).
        anom_max: User-defined anomaly max (heatmap end).
        tile: Tile size [h,w]; enables tiling preprocessing.
        stride: Stride [h,w]; required when tile is set.
        overlap_mode: Tile blend mode ("average", "max", "cosine", "linear", "gaussian").
        scale_mode: Tile scaling ("padding" or "interpolation").
        limit: If set, process only the first N images.
    """
    log = getattr(model, "logger", logger)

    directory_path = Path(images_path)
    images = list(directory_path.rglob("*.png")) + list(directory_path.rglob("*.jpg"))
    if limit is not None:
        images = images[:limit]
    log.info(f"{len(images)} images from {images_path}")
    if not images:
        return

    out_path = annot_dir
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    steps = []
    if tile is not None:
        if stride is None:
            raise ValueError("Must provide stride when using tiling")
        steps = [
            {
                "type": "tile",
                "configuration": {"tile_size": tile, "stride": stride, "overlap_mode": overlap_mode, "scale_mode": scale_mode},
            }
        ]
    preprocessor = Preprocessor()
    reconstructor = Reconstructor()

    proctime = []
    img_all, anom_all, fname_all, path_all = [], [], [], []
    for image_path in images:
        log.debug(f"Processing image: {image_path}.")
        image_path = str(image_path)
        img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        t0 = time.time()
        if steps:
            tiles, history = preprocessor.preprocess([img], steps)
            ad_maps = model.predict(tiles)
            anom_map = reconstructor.reconstruct_images(ad_maps, history)[0]
            if isinstance(anom_map, torch.Tensor):
                anom_map = anom_map.cpu().numpy()
        else:
            anom_map = model.predict([img])[0]
        proctime.append(time.time() - t0)
        fname = os.path.split(image_path)[1]
        if not steps:
            h, w = model.image_size
            img = pipeline_utils.resize_image(img, H=h, W=w)
        img_all.append(img)
        anom_all.append(anom_map)
        fname_all.append(fname)
        path_all.append(image_path)

    tp_print = None
    alpha_hat = loc_hat = beta_hat = None
    threshold = quantile_patch = quantile_sample = None

    if generate_stats:
        log.info("Computing anomaly score PDF for all data.")
        anom_sq = np.squeeze(np.array(anom_all))
        data = np.ravel(anom_sq)
        eps = 1e-6
        alpha_hat, loc_hat, beta_hat = gamma.fit(data + eps, floc=0)
        x = np.linspace(min(data), max(data), 1000)
        pdf_fitted = gamma.pdf(x, alpha_hat, loc=loc_hat, scale=beta_hat)
        plt.hist(data, bins=100, density=True, alpha=0.7, label="Observed Data")
        plt.plot(x, pdf_fitted, "r-", label="Fitted Gamma")
        plt.legend()
        plt.savefig(os.path.join(annot_dir, "gamma_pdf_fit.png"))
        max_data = max(data)
        threshold = np.linspace(min(data), max_data, 10)
        quantile_patch = 1 - gamma.cdf(threshold, alpha_hat, loc=loc_hat, scale=beta_hat)
        while quantile_patch.min() < MINIMUM_QUANT:
            log.warning(f"Patch quantile saturated with max anomaly score: {max_data}, reducing to {max_data / 2}")
            max_data = max_data / 1.2
            threshold = np.linspace(min(data), max_data, 10)
            quantile_patch = 1 - gamma.cdf(threshold, alpha_hat, loc=loc_hat, scale=beta_hat)
        quantile_patch_str = ["{:.{}e}".format(item * 100, 2) for item in np.squeeze(quantile_patch).tolist()]
        quantile_patch_str = ["Prob of Patch Defect"] + quantile_patch_str
        quantile_sample_str = ["Prob of Sample Defect"]
        quantile_sample = []
        for t in threshold:
            ind = np.where(anom_sq > t)
            ind_u = np.unique(ind[0])
            percent = len(ind_u) / len(fname_all)
            quantile_sample.append(percent)
            quantile_sample_str.append("{:.{}e}".format(percent * 100, 2))

        quantile_sample = np.array(quantile_sample)
        threshold_str = ["{:.{}e}".format(item, 2) for item in np.squeeze(threshold).tolist()]
        threshold_str = ["Threshold"] + threshold_str

        tp = [threshold_str, quantile_patch_str, quantile_sample_str]
        tp_print = tabulate(tp, tablefmt="grid")
        log.info("Threshold options:\n" + tp_print)

    if annotate_inputs:
        if anom_threshold is None and generate_stats:
            anom_threshold = gamma.ppf(0.5, alpha_hat, loc=loc_hat, scale=beta_hat)
            log.info(f"Anomaly patch threshold for 50% patch failure rate:{anom_threshold}")
        if anom_max is None and generate_stats:
            p_sample_target = 0.03
            if p_sample_target > quantile_sample.min():
                p_target = _find_p(threshold, quantile_patch, quantile_sample, p_sample_target)
                anom_max = gamma.ppf(1 - p_target, alpha_hat, loc=loc_hat, scale=beta_hat)
                log.info(f"Anomaly max set to 97 percentile:{anom_max}")
            else:
                anom_max = threshold.max()
                log.warning(
                    f"Anomaly patch max set to minimum discernable value: {anom_max} due to vanishing gradient in the patch quantile.  \
                        Sample failure rate: {quantile_sample.min() * 100:.2e}"
                )

        results = zip(img_all, anom_all, fname_all)
        _plot_fig(results, annot_dir, err_thresh=anom_threshold, err_max=anom_max)

    means = np.array([anom.mean() for anom in anom_all])
    maxs = np.array([anom.max() for anom in anom_all])
    stds = np.array([np.std(anom) for anom in anom_all])

    idx = np.argsort(maxs)[::-1]
    maxs = maxs[idx]
    means = means[idx]
    stds = stds[idx]
    fname_all = np.array(fname_all)[idx]

    with open(os.path.join(annot_dir, "stats.csv"), "w") as csvfile:
        fieldnames = ["fname", "mean", "max", "std"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in zip(fname_all, means, maxs, stds):
            writer.writerow({f: d for f, d in zip(fieldnames, row)})

    if proctime:
        proctime = np.asarray(proctime)
        log.info(f"Min Proc Time: {proctime.min()}")
        log.info(f"Max Proc Time: {proctime.max()}")
        log.info(f"Avg Proc Time: {proctime.mean()}")
        log.info(f"Median Proc Time: {np.median(proctime)}")
    log.info(f"Test results saved to {out_path}")
    if tp_print is not None:
        log.info("Threshold options:\n" + tp_print)
