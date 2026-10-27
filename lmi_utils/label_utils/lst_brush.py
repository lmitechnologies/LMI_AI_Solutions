"""Label Studio brush RLE: the encoding of a `brushlabels` result, an RGBA image flattened to bytes."""

from typing import List

import numpy as np

# bit widths a run length may be written with; LS writes the same table into every header
RLE_SIZES = (3, 4, 8, 16)
WORD_SIZE = 8


def _decode_values(rle: List[int]) -> np.ndarray:
    bits = "".join(f"{b:08b}" for b in rle)
    pos = 0

    def read(n):
        nonlocal pos
        value = int(bits[pos : pos + n], 2)
        pos += n
        return value

    num = read(32)
    word_size = read(5) + 1
    rle_sizes = [read(4) + 1 for _ in range(4)]
    out = np.zeros(num, dtype=np.uint8)
    i = 0
    while i < num:
        is_run = read(1)
        j = i + 1 + read(rle_sizes[read(2)])
        if is_run:
            out[i:j] = read(word_size)
        else:
            for k in range(i, j):
                out[k] = read(word_size)
        i = j
    return out


def decode_mask(rle: List[int], height: int, width: int) -> np.ndarray:
    """Decode a Label Studio brush RLE to a boolean mask of shape (height, width)."""
    return _decode_values(rle).reshape(height, width, 4)[:, :, 3] > 128


def encode_mask(mask: np.ndarray) -> List[int]:
    """Encode a 2D mask (nonzero = foreground) as a Label Studio brush RLE."""
    values = np.repeat((np.asarray(mask) > 0).ravel().astype(np.uint8) * 255, 4)
    starts = np.concatenate(([0], np.flatnonzero(values[1:] != values[:-1]) + 1))
    lengths = np.diff(np.append(starts, len(values)))

    parts = [f"{len(values):032b}", f"{WORD_SIZE - 1:05b}"] + [f"{s - 1:04b}" for s in RLE_SIZES]
    for start, length in zip(starts, lengths):
        value = int(values[start])
        while length > 0:
            n = int(min(length, 1 << RLE_SIZES[-1]))
            idx = next(k for k, s in enumerate(RLE_SIZES) if n <= 1 << s)
            parts.append(f"1{idx:02b}{n - 1:0{RLE_SIZES[idx]}b}{value:0{WORD_SIZE}b}")
            length -= n
    bits = "".join(parts)
    bits += "0" * (-len(bits) % 8)
    return np.packbits(np.frombuffer(bits.encode(), dtype=np.uint8) - ord("0")).tolist()
