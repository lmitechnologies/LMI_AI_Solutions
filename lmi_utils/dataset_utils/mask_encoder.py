import numpy as np
from pycocotools import mask as maskUtils


def mask2rle(mask: np.ndarray) -> str:
    mask = mask.astype(np.uint8)
    mask = np.asfortranarray(mask)
    rle = maskUtils.encode(mask)
    return rle.get('counts').decode('ascii')


def rle2mask(rle: str, h: int = 0, w: int = 0) -> np.ndarray:
    return maskUtils.decode({
        'counts': rle.encode('ascii'),
        'size': [h, w]
    })