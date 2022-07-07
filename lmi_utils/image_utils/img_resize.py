import numpy as np
import cv2

def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
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

    resized = cv2.resize(image, (width,height), interpolation=inter)


    return resized