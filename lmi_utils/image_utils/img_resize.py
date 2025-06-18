import numpy as np
import cv2
import os
import logging
import argparse
from gadget_utils.pipeline_utils import fit_array_to_size
from system_utils.path_utils import get_relative_paths

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def is_cuda_cv(): # 1 == using cuda, 0 = not using cuda
    try:
        count = cv2.cuda.getCudaEnabledDeviceCount()
        if count > 0:
            return True
        else:
            return False
    except:
        return False

def resize_and_pad(image, width=None, height=None, maintain_aspect_ratio=False):
    h, w = image.shape[:2]
    th, tw  = height, width
    
    if (tw is None and th is None) or (tw == w and th == h):
        th, tw = h, w
        im_out = image
    else:
        if maintain_aspect_ratio:
            scale = min(th / h, tw / w)
            tw = np.int32(scale * w)
            th = np.int32(scale * h)
            im_out = resize(image, width=tw, height=th) 
            if width is not None and height is not None:
                im_out, pad_l, _, pad_t, _ = fit_array_to_size(im_out, width, height)           
        else:    
            if tw is None:
                tw = w
                im_out = resize(image, height=th)
            elif th is None:
                th = h
                im_out = resize(image, width=tw)
            else:
                im_out = resize(image, width=tw, height=th)
    return im_out


def resize(image, width=None, height=None, device='cpu', inter=cv2.INTER_AREA):
    '''
    DESCRIPTION: 
        resizes images, preserving aspect ratio along argument free dimension
    ARGS:
        image: image np array
        width: desired width
        height: desired height
        inter: interpolation method
    '''
    if width == 0:
        width = None
    if height == 0:
        height = None
    
    if height == None and width == None:
        return image
    
    (h, w) = image.shape[:2]
    
    if h == height and width == width:
        return image

    if (height is None) and (width is None):
        return image
    if (height is None) and (width is not None):
        ratio = width / np.float32(w)
        height=np.int32(h * ratio)
    elif (width is None) and (height is not None):
        ratio = height / np.float32(h)
        width = np.int32(w * ratio)
    else:
        pass

    if device=='gpu':
        if not is_cuda_cv():
            device='cpu'

    if device=='gpu':
        src = cv2.cuda_GpuMat()
        src.upload(image)
        dest = cv2.cuda.resize(src, (width,height), interpolation=inter)
        resized=dest.download()      
    else:
        resized = cv2.resize(image, (width,height), interpolation=inter)

    return resized


def img_resize(input_path, output_path, width=None, height=None, recursive=False, maintain_aspect_ratio=False):
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
        raise Exception('Input path is not a directory')

    files = get_relative_paths(input_path, recursive)
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    out_w = width if width else 'w'
    out_h = height if height else 'h'
    
    for file in files:
        image = cv2.imread(os.path.join(input_path, file))
        resized = resize_and_pad(image=image, width=width, height=height, maintain_aspect_ratio=maintain_aspect_ratio)
        
        fname = os.path.basename(file)
        outname = fname.replace(os.path.splitext(file)[1], '.png')
        outname = outname.replace('.png', f'_resize_{out_w}x{out_h}.png')
        
        logger.debug(f'Writing {outname}')
        
        outp = os.path.join(output_path, os.path.dirname(file))
        if not os.path.exists(outp):
            os.makedirs(outp)
        
        cv2.imwrite(os.path.join(outp, outname), resized)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-i','--input_path', required=True, help='the path to images')
    ap.add_argument('-o','--output_path', required=True)
    ap.add_argument('--width', type=int, default=None)
    ap.add_argument('--height',type=int, default=None)
    ap.add_argument('--recursive', action='store_true', help='process images recursively')
    ap.add_argument(
        '--par', '-par', action='store_true',
        help='Maintain aspect ratio when resizing and pad when needed.'
    )
    args = vars(ap.parse_args())

    inpath=args['input_path']
    outpath=args['output_path']
    height=args['height']
    width=args['width']
    recursive=args['recursive']
    maintain_aspect_ratio=args['par']
    
    img_resize(
        input_path=inpath, 
        output_path=outpath, 
        width=width, 
        height=height, 
        recursive=recursive, 
        maintain_aspect_ratio=maintain_aspect_ratio
    )

if __name__=='__main__':
    main()