import argparse
import logging
import os

import cv2
import numpy as np

from lmi_utils.gadget_utils.pipeline_utils import fit_im_to_size, resize_image
from lmi_utils.system_utils.path_utils import get_relative_paths

logger = logging.getLogger(__name__)


def is_cuda_cv():  # 1 == using cuda, 0 = not using cuda
    try:
        count = cv2.cuda.getCudaEnabledDeviceCount()
        if count > 0:
            return True
        else:
            return False
    except Exception:
        return False


def resize_and_pad(image, width=None, height=None, preserve_aspect=False, **kwargs):
    """Resize a single image (and optionally letterbox-pad it) to a target size.

    Args:
        image: HW or HWC image (numpy array or torch tensor).
        width (int, optional): target width. Defaults to current width.
        height (int, optional): target height. Defaults to current height.
        preserve_aspect (bool): if True, scale to fit preserving aspect ratio and pad to (width, height).

    kwargs:
        mode (str): interpolation mode. Default "bilinear".
        return_operators (bool): if True, also return a history list (new schema).
        operators (list): seed history list. ``return_operators=True`` returns
            ``seed + new_entries``. Each entry is::

                {"type": "resize", "metadata": [{"src_size": [w, h], "dst_size": [w, h], "pad"?: [L, R, T, B]}]}
                {"type": "pad",    "metadata": [{"pad": [L, R, T, B]}]}

    Returns:
        Image (always), and history list when ``return_operators=True``.
    """
    h0, w0 = image.shape[:2]
    tw = width if width is not None else w0
    th = height if height is not None else h0
    operators = list(kwargs.get("operators", []))
    mode = kwargs.get("mode", "bilinear")

    if tw == w0 and th == h0:
        im_out = image
    else:
        if preserve_aspect:
            scale = min(th / h0, tw / w0)
            w1 = int(scale * w0)
            h1 = int(scale * h0)
            im_out = resize_image(image, W=w1, H=h1, mode=mode)
            entry = {"type": "resize", "metadata": [{"src_size": [w0, h0], "dst_size": [w1, h1]}]}
            if w1 != tw or h1 != th:
                im_out, pad_l, pad_r, pad_t, pad_b = fit_im_to_size(im_out, tw, th)
                entry["metadata"][0]["pad"] = [pad_l, pad_r, pad_t, pad_b]
            operators.append(entry)
        else:
            im_out = resize_image(image, W=tw, H=th, mode=mode)
            operators.append({"type": "resize", "metadata": [{"src_size": [w0, h0], "dst_size": [tw, th]}]})

    if kwargs.get("return_operators", False) is True:
        return im_out, operators
    return im_out


def resize(image, width=None, height=None, device="cpu", inter=cv2.INTER_AREA):
    """
    DESCRIPTION:
        resizes images, preserving aspect ratio along argument free dimension
    ARGS:
        image: image np array
        width: desired width
        height: desired height
        inter: interpolation method
    """
    if width == 0:
        width = None
    if height == 0:
        height = None

    if height is None and width is None:
        return image

    (h, w) = image.shape[:2]

    if h == height and width == width:
        return image

    if (height is None) and (width is None):
        return image
    if (height is None) and (width is not None):
        ratio = width / np.float32(w)
        height = np.int32(h * ratio)
    elif (width is None) and (height is not None):
        ratio = height / np.float32(h)
        width = np.int32(w * ratio)
    else:
        pass

    if device == "gpu":
        if not is_cuda_cv():
            device = "cpu"

    if device == "gpu":
        src = cv2.cuda_GpuMat()
        src.upload(image)
        dest = cv2.cuda.resize(src, (width, height), interpolation=inter)
        resized = dest.download()
    else:
        resized = cv2.resize(image, (width, height), interpolation=inter)

    return resized


def img_resize(
    input_path,
    output_path,
    width=None,
    height=None,
    recursive=False,
    maintain_aspect_ratio=False,
):
    """
    Resize images in the input path and save them to the output path.

    Args:
        input_path (str): Path to the input images.
        output_path (str): Path to save resized images.
        width (int, optional): Desired width of the resized images. Defaults to None.
        height (int, optional): Desired height of the resized images. Defaults to None.
        recursive (bool, optional): Process images recursively. Defaults to False.
        maintain_aspect_ratio (bool, optional): Maintain aspect ratio when resizing. Defaults to False.
    """
    if not os.path.isdir(input_path):
        raise Exception("Input path is not a directory")

    files = get_relative_paths(input_path, recursive)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    out_w = width if width else "w"
    out_h = height if height else "h"

    for file in files:
        image = cv2.imread(os.path.join(input_path, file))
        resized = resize_and_pad(image=image, width=width, height=height, preserve_aspect=maintain_aspect_ratio)

        fname = os.path.basename(file)
        outname = fname.replace(os.path.splitext(file)[1], ".png")
        outname = outname.replace(".png", f"_resize_{out_w}x{out_h}.png")

        logger.debug(f"Writing {outname}")

        outp = os.path.join(output_path, os.path.dirname(file))
        if not os.path.exists(outp):
            os.makedirs(outp)

        cv2.imwrite(os.path.join(outp, outname), resized)


def main():
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input_path", required=True, help="the path to images")
    ap.add_argument("-o", "--output_path", required=True)
    ap.add_argument("--width", type=int, default=None)
    ap.add_argument("--height", type=int, default=None)
    ap.add_argument("--recursive", action="store_true", help="process images recursively")
    ap.add_argument(
        "--par",
        "-par",
        action="store_true",
        help="Maintain aspect ratio when resizing and pad when needed.",
    )
    args = vars(ap.parse_args())

    inpath = args["input_path"]
    outpath = args["output_path"]
    height = args["height"]
    width = args["width"]
    recursive = args["recursive"]
    maintain_aspect_ratio = args["par"]

    img_resize(
        input_path=inpath,
        output_path=outpath,
        width=width,
        height=height,
        recursive=recursive,
        maintain_aspect_ratio=maintain_aspect_ratio,
    )


if __name__ == "__main__":
    main()
