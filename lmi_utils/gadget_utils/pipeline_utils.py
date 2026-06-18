import glob
import json
import logging
import os
import random
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
import torch
from torch.nn import functional as F

BLACK = (0, 0, 0)
TWO_TO_FIFTEEN = 2**15

logger = logging.getLogger(__name__)


@torch.inference_mode()
def resize_image(im, W=None, H=None, mode="bilinear"):
    """
    Args:
        im(np array | torch.tensor): the image of the shape (H,W) or (H,W,C)
        W(int): width
        H:(int): Height
        mode(str): 'nearest' | 'linear' | 'bilinear' | 'bicubic' | 'trilinear' | 'area' | 'nearest-exact'. Default: 'bilinear'
    """
    if W is None and H is None:
        return im

    # get the target width and height
    h, w = im.shape[:2]
    if W is None:
        W = int(w * H / h)
    elif H is None:
        H = int(h * W / w)

    # convert to tensor
    is_numpy = isinstance(im, np.ndarray)
    if is_numpy:
        im = torch.from_numpy(im)

    # deal with 1 channel image
    one_channel = im.ndim == 2
    if one_channel:
        im = im.unsqueeze(-1)

    # deal with integer image
    dtype = im.dtype
    is_fp = im.is_floating_point()
    if not is_fp:
        im = im.float()

    im2 = F.interpolate(im.permute(2, 0, 1).unsqueeze(0), size=(H, W), mode=mode)
    im2 = im2.squeeze(0).permute(1, 2, 0)

    # back to integer image
    if not is_fp:
        im2 = im2.to(dtype)

    # back to 1 channel
    if one_channel:
        im2 = im2.squeeze(-1)

    return im2.numpy() if is_numpy else im2


def _center_pad_amounts(target, cur):
    """
    Split the signed size delta ``target - cur`` into centered (low, high) pad amounts.
    Positive amounts pad, negative amounts crop. Matches F.pad's (begin, end) convention.
    """
    n = abs(target - cur)
    lo = n // 2
    hi = n - lo
    if target < cur:
        lo, hi = -lo, -hi
    return lo, hi


@torch.inference_mode()
def fit_im(im, pad_ops, value=0):
    """
    pad/crop the image using the pad_ops. This is the shared padding primitive;
    a positive op pads that edge, a negative op crops it (F.pad semantics).
    Args:
        im(np.array or torch.Tensor): the image of the shape (H,W) or (H,W,C)
        pad_ops(list): [pad_left, pad_right, pad_top, pad_bottom]
        value(int): the value to pad
    Returns:
        im(np.array or torch.Tensor): the padded/cropped image, same type as input
    """
    is_numpy = isinstance(im, np.ndarray)
    if is_numpy:
        im = torch.from_numpy(im)

    # deal with 1 channel image
    one_channel = im.ndim == 2
    if one_channel:
        im = im.unsqueeze(-1)

    # convert to CHW format
    im = im.permute(2, 0, 1)

    # pad/crop (negative ops crop)
    pad_L, pad_R, pad_T, pad_B = pad_ops
    im = F.pad(im, (pad_L, pad_R, pad_T, pad_B), value=value)

    # convert back to HWC format
    im = im.permute(1, 2, 0)

    # back to 1 channel
    if one_channel:
        im = im.squeeze(-1)

    if is_numpy:
        im = im.numpy()
    return im


def fit_im_to_size(im, W=None, H=None, value=0):
    """
    pad/crop the image to the size [W,H], centering the original content.
    Args:
        im(np.array or torch.Tensor): the image of the shape (H,W) or (H,W,C)
        W(int): the target width. If None, the width will not be changed
        H(int): the target height. If None, the height will not be changed
        value(int): the value to pad
    Returns:
        im(np.array or torch.Tensor): the padded/cropped image, same type as input
        pad_l(int): pixels padded to left (negative if cropped)
        pad_r(int): pixels padded to right (negative if cropped)
        pad_t(int): pixels padded to top (negative if cropped)
        pad_b(int): pixels padded to bottom (negative if cropped)
    """
    h, w = im.shape[:2]
    pad_L, pad_R = _center_pad_amounts(w if W is None else W, w)
    pad_T, pad_B = _center_pad_amounts(h if H is None else H, h)
    im = fit_im(im, (pad_L, pad_R, pad_T, pad_B), value=value)
    return im, pad_L, pad_R, pad_T, pad_B


def fit_array_to_size(im, W=None, H=None, value=0):
    """Backward-compatible numpy alias of fit_im_to_size; see that function."""
    return fit_im_to_size(im, W=W, H=H, value=value)


def uint16_to_int16(profile):
    """
    convert uint16 profile image to int16
    """
    is_numpy = isinstance(profile, np.ndarray)
    if is_numpy:
        profile = torch.from_numpy(profile)

    if profile.dtype != torch.uint16:
        raise Exception(f"input should be uint16, got {profile.dtype}")

    profile = profile.to(torch.int32) - torch.tensor(TWO_TO_FIFTEEN, dtype=torch.int32)
    profile = profile.to(torch.int16)
    return profile.numpy() if is_numpy else profile


@torch.inference_mode()
def profile_to_3d(profile, resolution, offset):
    """
    convert profile image to 3d sensor space

    Args:
        profile(np array | tensor): the profile image
        resolution(tuple): (x_resolution, y_resolution, z_resolution)
        offset(tuple): (x_offset, y_offset, z_offset)
    Returns:
        X: the x coordinates in 3d space, same shape as profile
        Y: the y coordinates in 3d space, same shape as profile
        Z: the z coordinates in 3d space, same shape as profile
        mask: the mask of the profile image to remove background
    """
    is_numpy = isinstance(profile, np.ndarray)

    # convert to tensor
    if is_numpy:
        profile = torch.from_numpy(profile)
    resolution = torch.from_numpy(np.array(resolution)).to(profile.device)
    offset = torch.from_numpy(np.array(offset)).to(profile.device)

    if profile.dtype != torch.int16:
        raise Exception(f"profile.dtype should be int16, got {profile.dtype}")

    h, w = profile.shape[:2]
    x1, y1 = 0, 0
    x2, y2 = w, h
    mask = profile != -TWO_TO_FIFTEEN
    x_range = torch.arange(x1, x2, device=profile.device)
    y_range = torch.arange(y1, y2, device=profile.device)
    xx, yy = torch.meshgrid(x_range, y_range, indexing="xy")
    X = offset[0] + xx * resolution[0]
    Y = offset[1] + yy * resolution[1]
    Z = offset[2] + profile * resolution[2]

    if is_numpy:
        X = X.numpy()
        Y = Y.numpy()
        Z = Z.numpy()
        mask = mask.numpy()
    return X, Y, Z, mask


def pts_to_3d(pts, profile, resolution, offset):
    """
    convert list of 2d pixel locations to 3d sensor space

    Args:
        pts(numpy | tensor): array of (x,y) points, with shape of Nx2
        profile(same type as pts): the profile image
        resolution(tuple): (x_resolution, y_resolution, z_resolution)
        offset(tuple): (x_offset, y_offset, z_offset)
    """
    if type(pts) is not type(profile):
        raise Exception(f"pts and profile should have the same type, got {type(pts)} and {type(profile)}")

    is_numpy = isinstance(pts, np.ndarray)
    if is_numpy:
        pts = torch.from_numpy(pts)
        profile = torch.from_numpy(profile)
    offset = torch.from_numpy(np.array(offset)).to(pts.device)
    resolution = torch.from_numpy(np.array(resolution)).to(pts.device)

    if pts.device != profile.device:
        raise Exception(f"device of pts and profile should be the same, got {pts.device} and {profile.device}")
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise Exception(f"shape of pts should be Nx2, got {pts.shape}")
    if profile.dtype != torch.int16:
        raise Exception(f"profile.dtype should be int16, got {profile.dtype}")

    pts = pts.to(torch.int32)
    Xs = pts[:, 0]
    Ys = pts[:, 1]

    nx = offset[0] + Xs * resolution[0]
    ny = offset[1] + Ys * resolution[1]
    nz = offset[2] + profile[Ys, Xs] * resolution[2]
    xyz = torch.stack([nx, ny, nz], dim=1)

    return xyz.numpy() if is_numpy else xyz


def plot_one_box(
    box,
    img,
    mask=None,
    mask_threshold: float = 0.0,
    color=None,
    label=None,
    line_thickness=None,
    hide_bbox=False,
):
    """
    Plots one bounding box and mask (optinal) on image img, this function comes from YoLov5 project.
    Args:
        box:    a box likes [x1,y1,x2,y2]
        img:    a opencv image object in BGR format
        mask:   a binary mask for the box
        color:  color to draw rectangle, such as (0,255,0)
        label:  str
        line_thickness: int
    Returns:
        no return
    """
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]

    box = np.array(box, dtype=int)
    if box.shape != (4,):
        raise Exception(f"box should be the shape of (4,), but got {box.shape}")
    if torch.is_tensor(mask):
        mask = mask.cpu().numpy()

    x1, y1, x2, y2 = box
    c1, c2 = (x1, y1), (x2, y2)
    if not hide_bbox:
        cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if mask is not None:
        # mask *= 255
        m = mask > mask_threshold
        blended = (0.4 * np.array(color, dtype=float) + 0.6 * img[m]).astype(np.uint8)
        img[m] = blended
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 4, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(
            img,
            label,
            (c1[0], c1[1] - 2),
            0,
            tl / 4,
            [225, 255, 255],
            thickness=tf,
            lineType=cv2.LINE_AA,
        )


def plot_one_rbox(box, img, color=None, label=None, line_thickness=None, hide_bbox=False):
    """
    Plots one bounding rotated bbox on image img
    Args:
        box:    a box likes [[x,y],[x,y],[x,y],[x,y]]
        img:    a opencv image object in BGR format
        mask:   a binary mask for the box
        color:  color to draw polygon, such as (0,255,0)
        label:  str
        line_thickness: int
    Returns:
        no return
    """
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]

    box = np.array(box, dtype=int)
    if box.shape != (4, 2):
        raise Exception(f"box should be the shape of 4x2, but got {box.shape}")

    if not hide_bbox:
        cv2.polylines(img, [box], isClosed=True, color=color, thickness=tl)

    if label:
        highest_point = min(box, key=lambda point: point[1])
        text_position = (highest_point[0], highest_point[1] - 10)

        if text_position[1] < 0:  # If the text would be outside the image, move it below the lowest point instead
            lowest_point = max(box, key=lambda point: point[1])
            text_position = (lowest_point[0], lowest_point[1] + 20)

        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 4, thickness=tf)[0]
        cv2.rectangle(
            img,
            text_position,
            (text_position[0] + t_size[0], text_position[1] - t_size[1] - 3),
            color,
            -1,
            cv2.LINE_AA,
        )  # filled
        cv2.putText(
            img,
            label,
            text_position,
            0,
            tl / 4,
            [225, 255, 255],
            thickness=tf,
            lineType=cv2.LINE_AA,
        )


_RECONSTRUCTORS = {}  # interpolation mode -> Reconstructor (image-revert resampling)


def _reconstructor(interpolation: str = "bilinear"):
    """Return a shared ``Reconstructor`` whose resize op reverts images with ``interpolation``."""
    recon = _RECONSTRUCTORS.get(interpolation)
    if recon is None:
        from lmi_utils.preprocess_utils.ops import ResizeOperation
        from lmi_utils.preprocess_utils.reconstructor import Reconstructor

        recon = Reconstructor()
        if interpolation != "bilinear":
            recon.register(ResizeOperation(image_mode=interpolation))
        _RECONSTRUCTORS[interpolation] = recon
    return recon


@torch.inference_mode()
def revert_mask_to_origin(mask, operations: list, interpolation: str = "bilinear"):
    """
    Revert a single mask image to original-image space using a preprocessing history.

    Args:
        mask: np.array or torch.Tensor, shape (H, W) or (H, W, C).
        operations: list of typed ``Meta`` records (one per preprocessing step), batch size 1.
        interpolation: resize-revert mode. Default is ``"bilinear"``.

    Returns:
        Mask reverted to original-image space, same type as input.
    """
    return _reconstructor(interpolation).reconstruct_images([mask], operations)[0]


@torch.inference_mode()
def revert_masks_to_origin(masks, operations: list, interpolation: str = "bilinear"):
    """
    Revert a stack of mask images (N, H, W) to original-image space. Batched form of :func:`revert_mask_to_origin`.
    """
    if len(masks) == 0:
        return []
    is_tensor = isinstance(masks[0], torch.Tensor)
    is_numpy = isinstance(masks, np.ndarray)

    results = [revert_mask_to_origin(m, operations, interpolation=interpolation) for m in masks]
    if is_tensor:
        return torch.stack(results)
    return np.stack(results) if is_numpy else results


def _transform_pts(pts, operations: list, *, reverse: bool, to_round: bool):
    """Route Nx2 points / Nx4 xyxy boxes through the Reconstructor's coord transform."""
    if not len(pts):
        return pts

    is_tensor = isinstance(pts, torch.Tensor)
    is_numpy = isinstance(pts, np.ndarray)
    if not is_tensor:
        pts = torch.from_numpy(pts) if is_numpy else torch.as_tensor(pts)

    if pts.ndim != 2 or pts.shape[1] not in (2, 4):
        raise Exception(f"pts should be Nx2 or Nx4, got shape: {pts.shape}")

    recon = _reconstructor()
    transform = recon.reconstruct_coordinates if reverse else recon.apply_coordinates
    if pts.shape[1] == 4:
        out = transform({"boxes": [pts]}, operations)["boxes"][0]
    else:
        out = transform({"segments": [[pts]]}, operations)["segments"][0][0]

    if to_round:
        out = out.round().clamp(min=0)
    if is_tensor:
        return out
    return out.cpu().numpy() if is_numpy else out.tolist()


@torch.inference_mode()
def revert_to_origin(pts, operations: list, round: bool = True, **kwargs):
    """
    Revert Nx2 points or Nx4 xyxy boxes to original-image space.

    Args:
        pts: torch.Tensor / np.ndarray / list of shape (N, 2) or (N, 4).
        operations: list of typed ``Meta`` records (one per preprocessing step), batch size 1.

    kwargs:
        round (bool): round and clamp output to non-negative integers. Default True.

    Returns:
        Same shape and type as input.
    """
    return _transform_pts(pts, operations, reverse=True, to_round=round)


@torch.inference_mode()
def apply_operations(pts, operations: list, round: bool = True):
    """
    Forward-apply preprocessing ops to original-space Nx2 points or Nx4 boxes.

    Inverse of :func:`revert_to_origin`. Dispatches to each op's ``apply_coords``.
    """
    return _transform_pts(pts, operations, reverse=False, to_round=round)


def convert_key_to_int(dt):
    """
    convert the class map <id, class name> to integer class id
    """
    return {int(k): dt[k] for k in dt}


def val_to_key(dt):
    return {dt[k]: k for k in dt}


def get_img_path_batches(batch_size, img_dir, fmt="png"):
    ret = []
    batch = []
    cnt_images = 0
    for root, _dirs, files in os.walk(img_dir):
        for name in files:
            if name.find(f".{fmt}") == -1:
                continue
            if len(batch) == batch_size:
                ret.append(batch)
                batch = []
            batch.append(os.path.join(root, name))
            cnt_images += 1
    logger.info(f"loaded {cnt_images} files")
    if len(batch) > 0:
        ret.append(batch)
    return ret


def get_gadget_img_batches(batch_size, profile_dir, intensity_dir, fmt="png"):
    profile_list = glob.glob(os.path.join(profile_dir, "*." + fmt))
    intensity_list = glob.glob(os.path.join(intensity_dir, "*." + fmt))

    profile_list.sort()
    intensity_list.sort()

    ret = []
    batch = []
    cnt_images = 0
    for profile, intensity in zip(profile_list, intensity_list):
        if len(batch) == batch_size:
            ret.append(batch)
            batch = []
        batch.append({"profile": profile, "intensity": intensity})
        cnt_images += 1
    logger.info(f"loaded {cnt_images} files")
    if len(batch) > 0:
        ret.append(batch)
    return ret


def get_gadget_inputs(path_im, path_surface_tar):
    """get gadget gocator inputs

    Args:
        path_im (str): path to a intensity image
        path_surface_tar (str): path to a gadget3d tar file

    Returns:
        dict: inputs to pipeline
    """
    assert path_surface_tar.endswith(".tar")

    im_name = os.path.basename(path_im)
    surface_name = os.path.basename(path_surface_tar)
    key1 = im_name.split(".")[0]
    key2 = surface_name.split(".")[0]
    assert key1 == key2

    # gocator returns a grayscale image
    im = cv2.imread(path_im)[:, :, 0]

    # create tmp fodler
    with tempfile.TemporaryDirectory() as tmp_folder:
        with tarfile.open(path_surface_tar, "r") as t:
            t.extractall(tmp_folder)
        # get surface data
        with open(os.path.join(tmp_folder, "metadata.json"), "r") as f:
            metadata = json.load(f)
        profile = cv2.imread(os.path.join(tmp_folder, "profile.png"), cv2.IMREAD_UNCHANGED)
        if profile.dtype == np.uint16:
            profile = profile.view(np.int16) + np.int16(-TWO_TO_FIFTEEN)

    out = {
        "image": {"pixels": im},
        "surface": {
            "profile": profile,
            "resolution": metadata["resolution"],
            "offset": metadata["offset"],
        },
    }
    return out


def load_pipeline_def(filepath):
    with open(filepath) as f:
        dt_all = json.load(f)
        li = dt_all["configs_def"]
        kwargs = {}
        for dt in li:
            kwargs[dt["name"]] = dt["default_value"]
    return kwargs


def _resolve_artifact_paths(model: Dict[str, Any], manifest_dir: Path) -> None:
    """Resolve relative artifact model_path values in-place against manifest_dir."""
    for _, artifact_data in model.get("artifacts", {}).items():
        raw_path = artifact_data.get("model_path")
        if raw_path:
            path_obj = Path(raw_path)
            if not path_obj.is_absolute():
                artifact_data["model_path"] = str((manifest_dir / path_obj).resolve())


def _validate_preprocessing_steps(models: List[Dict[str, Any]]) -> None:
    """Fail fast on bad preprocessing steps in a hand-authored static manifest."""
    from lmi_utils.preprocess_utils._parser import STEP_TYPES

    ignored = {"crop-to-label"}

    for model in models:
        role = model.get("model_role", "<unknown>")
        details = model.get("details") or {}
        steps = details.get("preprocessing") or []
        for i, step in enumerate(steps):
            op_type = step.get("type", "")
            if op_type in ignored:
                continue
            cfg_cls = STEP_TYPES.get(op_type)
            if cfg_cls is None:
                raise ValueError(f"Model role '{role}': preprocessing step at index {i} has unknown type '{op_type}'.")


def get_models_from_static_manifest(manifest_json_path: str, **kwargs):
    """
    Create models manifest from a static manifest json file.

    Args:
        manifest_json_path (str): path to the manifest JSON file.
        **kwargs: optional keyword arguments. Recognized:
            version (str): schema version of the manifest. Defaults to "3".
                v3 additionally validates preprocessing steps (rejecting unknown
                types) before building configs.
    """
    version = kwargs.get("version", "3")
    logger.info(f"Loading static manifest from {manifest_json_path} with schema version {version}")
    manifest_path = Path(manifest_json_path).resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    with open(manifest_path, "r") as f:
        models: List[Dict[str, Any]] = json.load(f)

    manifest: Dict[str, Any] = {}

    if version not in ["2", "3"]:
        raise ValueError(f"Unsupported static manifest version: {version}")

    manifest_v3 = version == "3"
    if manifest_v3:
        _validate_preprocessing_steps(models)

    keys_to_copy = (
        ["anomaly_size", "min_threshold", "max_threshold", "iou"]
        if manifest_v3
        else ["anomaly_size", "threshold_max", "threshold_min", "iou"]
    )
    classes_key = "classes" if manifest_v3 else "object_class"

    for model in models:
        role = model.get("model_role")
        if role is None:
            continue

        _resolve_artifact_paths(model, manifest_path.parent)

        # Create object configs
        model["configs"] = {}
        details = model.get("details", {})
        object_classes = details.get(classes_key, [])

        if object_classes:
            # Map config keys to their default values from details
            config_defaults = {
                "confidence": details.get("confidence_threshold", 0.5),
                "to-fail": details.get("to_fail", True),
            }
            if not manifest_v3:
                config_defaults["size"] = details.get("object_size", 1)

            # Generate the dictionary for each config key
            for config_key, default_val in config_defaults.items():
                model["configs"][config_key] = {cls: default_val for cls in object_classes}

        # Copy specific keys from details to configs
        for key in keys_to_copy:
            if key in details:
                model["configs"][key] = details[key]

        # Add to the manifest
        manifest[role] = model

    return manifest


def blur_mask(mask: np.ndarray, kernel_size: int, distance_based: bool = False):
    if not kernel_size:
        return mask

    if distance_based:
        blur_kernel_options = np.array([0, 3, 5])
        if kernel_size not in blur_kernel_options:
            kernel_size = blur_kernel_options[abs(blur_kernel_options - kernel_size).argmin()]
            logger.warning(
                f"Blur kernel size must be in {list(blur_kernel_options)} when not using "
                f"simple blur (Distance Transform); using {kernel_size} instead"
            )
        mask_bin = (mask > 0.5).astype(np.uint8)
        dist = cv2.distanceTransform(1 - mask_bin, cv2.DIST_L2, kernel_size)
        # convert distance to soft weights
        sigma = 5.0
        return np.exp(-(dist**2) / (2 * sigma**2))

    # Gaussian blur
    if not kernel_size % 2:
        logger.warning(f"Blur kernel size must be odd when using simple (Gaussian) blur; using {kernel_size + 1} instead")
        kernel_size += 1
    return cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)


def apply_ad_mask(err_map: np.ndarray, od_predictions: dict, mask_config: dict, class_names=None, defect_mask=None):
    DEFAULT_WEIGHT = 1
    DEFAULT_CONF_WEIGHT = True
    DEFAULT_ERODE_KERNEL = 0
    DEFAULT_BLUR_KERNEL = 11
    DEFAULT_REDUCE = True
    DEFAULT_SIMPLE_BLUR = True

    global_mult = mask_config["global_weight"]
    mask_config = mask_config["masking_params"]
    total_mask = np.zeros(err_map.shape)
    if class_names is not None:
        class_names = [c.lower() for c in class_names]

    # Resize defect_mask to err_map size
    H, W = err_map.shape[:2]
    if isinstance(defect_mask, np.ndarray):
        defect_mask = resize_image(defect_mask, W=W, H=H)

    for i, mask in enumerate(od_predictions["masks"]):
        fp_class = od_predictions["classes"][i].lower()
        # If class_names is defined, fp_class should be in class_names
        # (for using only specific classes from a model)
        if class_names is not None and fp_class not in class_names:
            continue
        cls_mask_config = mask_config.get(fp_class, {})
        use_conf_damp = cls_mask_config.get("weight_by_confidence", DEFAULT_CONF_WEIGHT)
        conf_damp = od_predictions["scores"][i] if use_conf_damp else 1
        weight = conf_damp * global_mult * cls_mask_config.get("weight", DEFAULT_WEIGHT)
        if not weight:
            continue

        erode_kernel = cls_mask_config.get("erode_kernel_size", DEFAULT_ERODE_KERNEL)
        blur_kernel = cls_mask_config.get("blur_kernel_size", DEFAULT_BLUR_KERNEL)
        mask_mult = cls_mask_config.get("multiply_by_mask", DEFAULT_REDUCE)
        simple_blur = cls_mask_config.get("simple_blur", DEFAULT_SIMPLE_BLUR)

        base_mask = resize_image(mask, W=W, H=H)
        if erode_kernel:
            mask_bin = (base_mask > 0.5).astype(np.uint8)
            kernel_size = abs(erode_kernel)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            size_transform = cv2.erode if erode_kernel > 0 else cv2.dilate
            base_mask = size_transform(mask_bin, kernel).astype(np.float32)

        fp_mask = np.clip(weight * blur_mask(base_mask, kernel_size=blur_kernel, distance_based=not simple_blur), 0, 1)
        if isinstance(defect_mask, np.ndarray):
            fp_mask = np.clip(fp_mask - defect_mask, 0, 1)
        if mask_mult:
            err_map *= 1 - fp_mask
            total_mask = 1 - (1 - total_mask) * (1 - fp_mask)
        else:
            err_map -= fp_mask
            total_mask += fp_mask

    total_mask = np.clip(total_mask, 0, 1)
    return err_map, total_mask


def masked_ad_predict(
    pipe, ad_inp, ad_model_role: str | np.ndarray, od_model_role: str, configs: dict, class_names=None, known_defects=None
):

    def get_od_predictions(model_role, inp):
        od_inp, ops = pipe.preprocess(model_role, [inp])
        conf = configs["models"][model_role]["configs"]["confidence"]
        od_predictions, time_info = pipe.models[model_role].predict(od_inp[0], conf)
        # Take first item from batch
        return {k: v[0] for k, v in pipe.revert_preprocess(od_predictions, ops).items()}

    ## Get OD predictions
    od_predictions = get_od_predictions(od_model_role, ad_inp)

    ## Parse err_map
    err_map = (
        pipe.models[ad_model_role].predict(ad_inp)[0] if isinstance(ad_model_role, str) else ad_model_role  # ad_role is err_map
    )
    if not od_predictions or not od_predictions["masks"].any():
        return (err_map, od_predictions, np.zeros_like(ad_inp, dtype=np.uint8))

    ## Parse known_defects (supported defect model_role, defect_names, defect_masks)
    class_names = set(class_names or configs["models"][od_model_role]["configs"]["confidence"])
    defect_mask = None
    if isinstance(known_defects, list):
        # Allow passing list of np.ndarray masks or list of defect names (str) to take from OD masks
        first_item = known_defects[0] if known_defects else None
        if isinstance(first_item, np.ndarray):  # Provide masks directly
            defect_mask = np.max(known_defects, axis=0)
        elif isinstance(first_item, str):  # Use masks from od_model_role
            known_defect_names = {x.lower() for x in known_defects}
            selected_masks = [
                mask for i, mask in enumerate(od_predictions["masks"]) if od_predictions["classes"][i].lower() in known_defect_names
            ]
            defect_mask = np.max(selected_masks, axis=0) if selected_masks else None
            class_names -= known_defect_names
    elif known_defects is not None:
        if isinstance(known_defects, np.ndarray):
            defect_mask = known_defects
        else:  # Interpret known_defects as a model_role
            assert known_defects in pipe.models, f"{known_defects} is not a valid model_role"
            assert known_defects != od_model_role, "Providing model known_defects as same role as od_model_role is not supported"
            defect_mask = get_od_predictions(known_defects, ad_inp)["masks"]

        defect_mask = np.asarray(defect_mask, dtype=np.float32)
        if not len(defect_mask):
            defect_mask = None
        elif len(defect_mask.shape) == 3:
            defect_mask = np.max(defect_mask, axis=0)

    masked_err_map, mask = apply_ad_mask(
        err_map, od_predictions, mask_config=configs[od_model_role], class_names=class_names, defect_mask=defect_mask
    )
    mask_img = cv2.cvtColor((mask * 255).astype("uint8"), cv2.COLOR_GRAY2RGB)
    return masked_err_map, od_predictions, mask_img


def masked_ad_annotate(pipe, img, ad_model_role, od_model_role, err_map, od_predictions, configs, color=(152, 251, 152)):
    fp_classes = configs["models"][od_model_role]["configs"]["confidence"].keys()
    ad_configs = configs["models"][ad_model_role]["configs"]
    err_threshold = ad_configs["min_threshold"]
    err_max = ad_configs["max_threshold"]
    annotated_img = pipe.models[ad_model_role].annotate(img, err_map, err_threshold, err_max)
    if not color:
        return annotated_img
    return pipe.models[od_model_role].annotate_image(od_predictions, annotated_img, {c: color for c in fp_classes})
