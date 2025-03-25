import numpy as np
import base64
import json
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def mask2rle(mask: np.ndarray):
    mask = mask.astype(int)
    vec = mask.flatten()
    nb = len(vec)
    starts = np.r_[0, np.flatnonzero(~np.isclose(vec[1:], vec[:-1], equal_nan=True)) + 1]
    lengths = np.diff(np.r_[starts, nb])
    values = vec[starts]
    assert len(starts) == len(lengths) == len(values)
    rle = []
    for start, length, val in zip(starts, lengths, values):
        if val == 0:
            continue
        rle += [str(start), length]
    rle = " ".join(map(str, rle))
    encoded = base64.b85encode(rle.encode('ascii')).decode('ascii')
    return encoded

def rle2mask(rle: str, h:int = 0, w: int = 0, label: int = 1) -> np.ndarray:
    rle_code = base64.b85decode(rle.encode('ascii')).decode('ascii')
    img_shape = (h, w)
    seq = rle_code.split()
    starts = np.array(list(map(int, seq[0::2])))
    lengths = np.array(list(map(int, seq[1::2])))
    ends = starts + lengths
    img = np.zeros((h * w,), dtype=np.uint8)
    for begin, end in zip(starts, ends):
        img[begin:end] = label
    return img.reshape(img_shape)