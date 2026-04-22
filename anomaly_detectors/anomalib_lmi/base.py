import json
import logging
import os
import subprocess
from abc import abstractmethod
from typing import List, Union

import cv2
import numpy as np
import torch
from torchvision.transforms import v2

import lmi_utils.gadget_utils.pipeline_utils as pipeline_utils
from anomaly_detectors.ad_core.ad_base import ADBase
from lmi_common.trt_engine import TRTEngine
from lmi_utils.preprocess_utils.preprocessor import Preprocessor
from lmi_utils.preprocess_utils.reconstructor import Reconstructor

MINIMUM_QUANT = 1e-12


def to_list(data):
    """convert to a two element list

    Args:
        data (int | list): a int or a two element list

    Returns:
        list: _description_
    """
    if isinstance(data, int):
        return [data] * 2
    if len(data) != 2:
        raise Exception(f"Must be a two element list, but got {data}")
    return list(data)


class Anomalib_Base(ADBase):
    logger = logging.getLogger("Anomalib Base")

    @abstractmethod
    def __init__(self) -> None:
        pass

    def _load_tensorrt_model(self, model_path: str) -> None:
        self.trt = TRTEngine(model_path, device=str(self.device))
        if len(self.trt._input_names) != 1:
            raise ValueError(f"Expected a single-input TRT engine, got inputs: {self.trt._input_names}")
        self.image_size = list(self.trt.input_shape[-2:])
        self.batch_size = self.trt.max_batch
        self.fp16 = self.trt.fp16
        self.inference_mode = "TRT"
        if not self.trt.is_dynamic:
            self.fixed_batch_size = self.trt.max_batch

    @torch.inference_mode()
    def preprocess(self, images) -> torch.Tensor:
        """Convert a list of HWC uint8 images to a batched [N,C,H,W] float tensor.

        Args:
            images: List of uint8 numpy arrays or torch tensors [H,W,C] or [H,W]

        Returns:
            Preprocessed tensor [N,C,H,W] float32 (or float16 if fp16)

        Raises:
            ValueError: If batch size exceeds TensorRT engine limit
        """
        tensors = []
        for image in images:
            img = self.from_numpy(image).float()
            tensors.append(img.permute((2, 0, 1)))  # [C,H,W]

        img = torch.stack(tensors) / 255.0  # [N,C,H,W]

        batch = img.shape[0]
        if self.inference_mode == "TRT" and batch > self.batch_size:
            raise ValueError(f"Batch size {batch} exceeds TensorRT engine max batch size {self.batch_size}")

        if self.inference_mode == "TRT" and (img.shape[2] != self.image_size[0] or img.shape[3] != self.image_size[1]):
            img = v2.Resize(self.image_size, antialias=True)(img)

        img = img.contiguous()
        return img.half() if self.fp16 else img

    def postprocess(self, output: torch.Tensor, return_numpy: bool = True) -> List[Union[np.ndarray, torch.Tensor]]:
        """Convert raw model output to a list of per-image anomaly maps.

        Args:
            output: Model output tensor [N,H,W] or [N,1,H,W].
            return_numpy: If True, return numpy arrays; otherwise return tensors.

        Returns:
            List of anomaly maps [H,W], one per input image.
        """
        output = output.squeeze(1) if output.ndim == 4 else output
        if return_numpy:
            output_np = output.cpu().numpy()
            return [np.squeeze(output_np[i]) for i in range(output_np.shape[0])]
        return [output[i].squeeze() for i in range(output.shape[0])]

    def warmup(self, input_hw=None):
        """Warm up model using a dummy zeros array.

        Args:
            input_hw: Input height and width as int (h==w) or [h, w]. Defaults to the model's built-in shape.
        """
        if input_hw is None:
            input_hw = self.image_size
        input_hw = to_list(input_hw)
        zeros = np.zeros(input_hw + [3], dtype=np.uint8)
        self.logger.info(f"Warming up model with input shape: {zeros.shape}")
        self.predict([zeros])

    def convert_to_onnx(self, export_path, opset_version=14):
        """
        Desc: Convert existing .pt file to onnx
        Args:
            - path to output .onnx file
            - opset_version: onnx version ID
        """
        # write metadata to export path
        json_file = os.path.join(os.path.dirname(export_path), "metadata.json")
        if hasattr(self, "pt_metadata"):
            with open(json_file, "w", encoding="utf-8") as metadata_file:
                json.dump(self.pt_metadata, metadata_file, ensure_ascii=False, indent=4)

        b, c = 1, 3
        h, w = self.image_size
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
            raise Exception("trt engine file must end with '.engine'")

        out_dir = os.path.dirname(out_engine_path)
        os.makedirs(out_dir, exist_ok=True)

        # run convert cmd
        cmd = [
            "trtexec",
            f"--onnx={onnx_path}",
            f"--saveEngine={out_engine_path}",
            f"--memPoolSize=workspace:{workspace}",
        ]
        if fp16:
            cmd.append("--fp16")
        subprocess.run(cmd, check=True)

        # check if metadata.json exists in the same directory as onnx_path
        onnx_dir = os.path.dirname(onnx_path)
        if os.path.isfile(f"{onnx_dir}/metadata.json"):
            cmd2 = [f"cp -sf {onnx_dir}/metadata.json {out_dir}"]
            subprocess.run(cmd2, shell=True)
        else:
            self.logger.warning(f"metadata.json not found in {onnx_dir}")

    def convert(self, model_path, export_path, fp16=True, convert_type="trt"):
        """
        Desc: Converts .onnx or .pt file to ONNX or TensorRT engine.

        Args:
            - model_path: model file path (.pt or .onnx)
            - export_path: output directory
            - fp16: use half precision for TRT conversion
            - convert_type: "onnx" to export ONNX only, "trt" to export TensorRT engine
        """
        if os.path.isfile(export_path):
            raise Exception("Export path should be a directory.")
        ext = os.path.splitext(model_path)[1]

        def pt_to_onnx():
            if ext != ".pt":
                raise ValueError(f"ONNX export requires a .pt input, got {ext}")
            self.logger.info("Converting pt to onnx...")
            onnx_path = os.path.join(export_path, "model.onnx")
            self.convert_to_onnx(onnx_path)
            self.logger.info(f"ONNX model saved at {onnx_path}")
            return onnx_path

        def onnx_to_trt(onnx_path):
            self.logger.info("Converting onnx to trt engine...")
            trt_path = os.path.join(export_path, "model.engine")
            self.convert_trt(onnx_path, trt_path, fp16)
            return trt_path

        if convert_type == "onnx":
            return pt_to_onnx()

        if convert_type == "trt":
            if ext == ".onnx":
                return onnx_to_trt(model_path)
            if ext == ".pt":
                return onnx_to_trt(pt_to_onnx())
            raise ValueError(f"TRT export requires a .pt or .onnx input, got {ext}")

        raise ValueError(f"Unknown convert_type: {convert_type!r}. Expected 'onnx' or 'trt'")

    def test(
        self,
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
            - tile: tile size [h,w]. If set, tiling is performed via Preprocessor before inference.
            - stride: stride size [h,w]. Required when tile is set.
            - overlap_mode: overlap blending for tiling, can be "average", "max", "cosine", "linear", "gaussian"
            - scale_mode: tile scaling mode, "padding" or "interpolation"
            - limit: if set, process only the first N images (useful for smoke tests)
        """
        import csv
        import time
        from pathlib import Path

        import matplotlib.pyplot as plt
        from scipy import interpolate
        from scipy.stats import gamma
        from tabulate import tabulate

        from anomaly_detectors.anomalib_lmi.ad_utils import plot_fig

        def find_p(thresh_array, p_patch_array, p_sample_array, p_sample_target):
            """
            Desc: Find the p-value that acheives the desired sample failure rate.  We start by estimating the threshold
                from the empiracal p_sample_array.  Then we use that threshold to estimate the corresponding p_patch.

            Args:
                - thresh_array: input threshold array
                - p_patch_array: corresponding p-value at the patch level (generated using gamma dist model)
                - p_sample_array: corresponding p-value at the sample level (generated empirically)
                - p_sample_target: desired sample level p-value
            """
            x1 = p_sample_array
            x2 = thresh_array
            x3 = p_patch_array
            # interpolation function to find threshold for a specified p_sample
            f1 = interpolate.interp1d(x1, x2)
            # estimate the threshold for p_sample_target
            thresh_target = f1(p_sample_target)
            # interpolation function to find p_patch for a specified threshold
            f2 = interpolate.interp1d(x2, x3)
            # find the threshold for the p_patch that corresponds to p_sample_target
            p_target = f2(thresh_target)
            return p_target

        # Input data
        directory_path = Path(images_path)
        images = list(directory_path.rglob("*.png")) + list(directory_path.rglob("*.jpg"))
        if limit is not None:
            images = images[:limit]
        self.logger.info(f"{len(images)} images from {images_path}")
        if not images:
            return

        # Output overhead
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
            self.logger.debug(f"Processing image: {image_path}.")
            image_path = str(image_path)
            img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
            t0 = time.time()
            if steps:
                tiles, history = preprocessor.preprocess([img], steps)
                ad_maps = self.predict(tiles)
                anom_map = reconstructor.reconstruct(ad_maps, history)[0]
                if isinstance(anom_map, torch.Tensor):
                    anom_map = anom_map.cpu().numpy()
            else:
                anom_map = self.predict([img])[0]
            proctime.append(time.time() - t0)
            fname = os.path.split(image_path)[1]
            if not steps:
                h, w = self.image_size
                img = pipeline_utils.resize_image(img, H=h, W=w)
            img_all.append(img)
            anom_all.append(anom_map)
            fname_all.append(fname)
            path_all.append(image_path)

        if generate_stats:
            # Compute & Validate pdf
            self.logger.info("Computing anomaly score PDF for all data.")
            anom_sq = np.squeeze(np.array(anom_all))
            data = np.ravel(anom_sq)
            # Fit gamma distribution to anomaly data across entire data set
            eps = 1e-6
            alpha_hat, loc_hat, beta_hat = gamma.fit(data + eps, floc=0)
            # Plot histogram and gamma dist fit
            x = np.linspace(min(data), max(data), 1000)
            pdf_fitted = gamma.pdf(x, alpha_hat, loc=loc_hat, scale=beta_hat)
            plt.hist(data, bins=100, density=True, alpha=0.7, label="Observed Data")
            plt.plot(x, pdf_fitted, "r-", label="Fitted Gamma")
            plt.legend()
            plt.savefig(os.path.join(annot_dir, "gamma_pdf_fit.png"))
            max_data = max(data)
            # Generate uniform anomaly threshold samples across available anomaly score range
            threshold = np.linspace(min(data), max_data, 10)
            # Determine percentage of failed parts for each threshold
            quantile_patch = 1 - gamma.cdf(threshold, alpha_hat, loc=loc_hat, scale=beta_hat)
            # Reduce threshold range when threshold values are too far into the tail of the gamma distribution (quantile goes to zero)
            while quantile_patch.min() < MINIMUM_QUANT:
                self.logger.warning(f"Patch quantile saturated with max anomaly score: {max_data}, reducing to {max_data / 2}")
                max_data = max_data / 1.2
                threshold = np.linspace(min(data), max_data, 10)
                quantile_patch = 1 - gamma.cdf(threshold, alpha_hat, loc=loc_hat, scale=beta_hat)
            # Extract patch level distribution table data
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
            # Print statistics
            tp_print = tabulate(tp, tablefmt="grid")
            self.logger.info("Threshold options:\n" + tp_print)

        if annotate_inputs:
            if anom_threshold is None and generate_stats:
                anom_threshold = gamma.ppf(0.5, alpha_hat, loc=loc_hat, scale=beta_hat)
                self.logger.info(f"Anomaly patch threshold for 50% patch failure rate:{anom_threshold}")
            if anom_max is None and generate_stats:
                # Sample target hard coded for 3% failure rate
                p_sample_target = 0.03
                if p_sample_target > quantile_sample.min():
                    # estimate p_patch from target p_sample_target
                    p_target = find_p(threshold, quantile_patch, quantile_sample, p_sample_target)
                    # find the anomaly score coresponding to that p_patch
                    anom_max = gamma.ppf(1 - p_target, alpha_hat, loc=loc_hat, scale=beta_hat)
                    self.logger.info(f"Anomaly max set to 97 percentile:{anom_max}")
                else:
                    anom_max = threshold.max()
                    self.logger.warning(
                        f"Anomaly patch max set to minimum discernable value: {anom_max} due to vanishing gradient in the patch quantile.  \
                            Sample failure rate: {quantile_sample.min() * 100:.2e}"
                    )

            results = zip(img_all, anom_all, fname_all)
            plot_fig(results, annot_dir, err_thresh=anom_threshold, err_max=anom_max)

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
        with open(os.path.join(annot_dir, "stats.csv"), "w") as csvfile:
            fieldnames = ["fname", "mean", "max", "std"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for data in zip(fname_all, means, maxs, stds):
                tmp_dict = {f: d for f, d in zip(fieldnames, data)}
                writer.writerow(tmp_dict)

        if proctime:
            proctime = np.asarray(proctime)
            self.logger.info(f"Min Proc Time: {proctime.min()}")
            self.logger.info(f"Max Proc Time: {proctime.max()}")
            self.logger.info(f"Avg Proc Time: {proctime.mean()}")
            self.logger.info(f"Median Proc Time: {np.median(proctime)}")
        self.logger.info(f"Test results saved to {out_path}")
        if generate_stats:
            # Repeat error table
            self.logger.info("Threshold options:\n" + tp_print)
