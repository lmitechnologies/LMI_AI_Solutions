import numpy as np
import cv2
import os
import logging

from image_utils.classifier_utils import get_im_relative_path

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
    (h, w) = image.shape[:2]

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

if __name__=='__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('-i','--input_path', required=True, help='the path to images')
    ap.add_argument('-o','--output_path', default='./')
    ap.add_argument('--width', type=int, default=None)
    ap.add_argument('--height',type=int, default=None)
    args = vars(ap.parse_args())

    inpath=args['input_path']
    outpath=args['output_path']
    height=args['height']
    width=args['width']
    
    if not os.path.isdir(inpath):
        raise Exception('Input path is not a directory')

    files = get_im_relative_path(inpath)
    
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    
    out_w = width if width else 'w'
    out_h = height if height else 'h'
    for file in files:
        image=cv2.imread(os.path.join(inpath,file))
        resized=resize(image,width,height)
        # change file name
        fname = os.path.basename(file)
        outname = fname.replace(os.path.splitext(file)[1],'.png')
        outname = outname.replace('.png',f'_resize_{out_w}x{out_h}.png')
        outp = os.path.join(outpath,os.path.dirname(file))
        if not os.path.exists(outp):
            os.makedirs(outp)
        cv2.imwrite(os.path.join(outp,outname),resized)
