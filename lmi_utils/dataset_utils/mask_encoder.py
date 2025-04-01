import numpy as np
from pycocotools import mask as maskUtils

# We want to use the coco mask format, but with row-major instead of column-major order.
# In order to keep the implemetation simple, we're just trasposing before/after encoding/decoding.
# This approach is somewhat inefficient; if it's ever an issue, we can provide our own implementation
# of the encoding/decoding functions.

def mask2rle(mask: np.ndarray) -> str:
    mask = mask.astype(np.uint8)
    mask = mask.T    
    mask = np.asfortranarray(mask)
    rle = maskUtils.encode(mask)
    return rle.get('counts').decode('ascii')


def rle2mask(rle: str, h: int = 0, w: int = 0) -> np.ndarray:
    return maskUtils.decode({
        'counts': rle.encode('ascii'),
        'size': [w, h]
    }).T
