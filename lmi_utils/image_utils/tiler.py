import json
import logging
from enum import Enum
from itertools import product
from math import ceil, floor
from pathlib import Path
from typing import NamedTuple, Tuple

import torch
from torch.nn import functional as F

logger = logging.getLogger()

GAUSSIAN_SIGMA = 0.5  # gaussian falloff width, as a fraction of the blend region; weight at a tile edge is exp(-2)


class ScaleMode(str, Enum):
    """Type of mode when upscaling image."""

    PADDING = "padding"
    INTERPOLATION = "interpolation"


class OverlapMode(str, Enum):
    """Type of blending mode for tile edges."""

    AVERAGE = "average"
    LINEAR = "linear"
    COSINE = "cosine"
    GAUSSIAN = "gaussian"
    MAX = "max"


def compute_new_edges(edges: list, tile_size: list, stride: list):
    def __compute_new_edge(edge, tile, stride):
        if edge <= tile:
            return tile
        if (edge - tile) % stride != 0:
            return tile + ceil((edge - tile) / stride) * stride
        return edge

    out_h = __compute_new_edge(edges[0], tile_size[0], stride[0])
    out_w = __compute_new_edge(edges[1], tile_size[1], stride[1])
    return out_h, out_w


def create_blend_mask(
    tile_size: list,
    stride: list,
    overlap_mode: OverlapMode = OverlapMode.AVERAGE,
    device="cpu",
    neighbours: Tuple[bool, bool, bool, bool] = (True, True, True, True),
) -> torch.Tensor:
    """Create a blending mask for tile transitions.

    Args:
        tile_size (list): [tile_h, tile_w]
        stride (list): [stride_h, stride_w]
        overlap_mode (OverlapMode): Type of blending to apply
        device: Device to create tensor on
        neighbours: whether another tile sits (top, left, bottom, right) of this one. An edge with no
            neighbour keeps full weight; tapering it leaves the image border with no weight to divide by.

    Returns:
        torch.Tensor: Blending mask of shape [tile_h, tile_w]
    """
    tile_h, tile_w = tile_size
    stride_h, stride_w = stride

    # Calculate overlap regions
    overlap_h = tile_h - stride_h
    overlap_w = tile_w - stride_w

    if overlap_h <= 0 and overlap_w <= 0:
        return torch.ones(tile_h, tile_w, device=device)

    mask = torch.ones(tile_h, tile_w, device=device)

    if overlap_mode == OverlapMode.AVERAGE:
        return mask

    if overlap_mode == OverlapMode.MAX:
        return mask

    # distance-based blending
    y_coords = torch.arange(tile_h, device=device).float()
    x_coords = torch.arange(tile_w, device=device).float()

    # create 2D grids
    y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing="ij")

    # Calculate distance from edges as 2D grids (not 1D vectors):
    # torch.minimum(y_blend, x_blend) below combines per-axis blends
    # elementwise over the full (tile_h, tile_w) surface, which only works
    # if both operands already have that shape.
    has_top, has_left, has_bottom, has_right = neighbours
    far = torch.full_like(y_grid, float(tile_h + tile_w))  # an edge with no neighbour must never be the nearest one
    y_dist_from_top = y_grid if has_top else far
    y_dist_from_bottom = tile_h - 1 - y_grid if has_bottom else far
    x_dist_from_left = x_grid if has_left else far
    x_dist_from_right = tile_w - 1 - x_grid if has_right else far

    # calculate minimum distance to any edge
    y_edge_dist = torch.minimum(y_dist_from_top, y_dist_from_bottom)
    x_edge_dist = torch.minimum(x_dist_from_left, x_dist_from_right)

    # blend over half the overlap, at least 1 px
    blend_region_h = max(1, overlap_h // 2)
    blend_region_w = max(1, overlap_w // 2)

    # the ramp is offset by a pixel so the outermost row and column still carry weight: on a 1 px overlap both
    # tiles sit on their own edge, and a ramp reaching 0 there leaves that seam with no weight to divide by
    # an axis without overlap (a single row or column, or stride == tile) has no seam to blend
    y_ramp = torch.clamp((y_edge_dist + 1) / (blend_region_h + 1), 0, 1) if overlap_h > 0 else torch.ones_like(y_grid)
    x_ramp = torch.clamp((x_edge_dist + 1) / (blend_region_w + 1), 0, 1) if overlap_w > 0 else torch.ones_like(x_grid)

    if overlap_mode == OverlapMode.LINEAR:
        mask = torch.minimum(y_ramp, x_ramp)

    elif overlap_mode == OverlapMode.COSINE:
        y_blend = 0.5 * (1 + torch.cos(torch.pi * (1 - y_ramp)))
        x_blend = 0.5 * (1 + torch.cos(torch.pi * (1 - x_ramp)))
        mask = torch.minimum(y_blend, x_blend)

    elif overlap_mode == OverlapMode.GAUSSIAN:
        # taper inside the overlap band only, like linear and cosine: a gaussian of the distance from the tile
        # centre underflows to zero over the whole tile once the overlap drops below about 15% of the tile
        y_blend = torch.exp(-((1 - y_ramp) ** 2) / (2 * GAUSSIAN_SIGMA**2))
        x_blend = torch.exp(-((1 - x_ramp) ** 2) / (2 * GAUSSIAN_SIGMA**2))
        mask = torch.minimum(y_blend, x_blend)

    return mask


def restore_dtype(image: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Cast a float working tensor back to ``dtype``.

    A plain ``.to()`` truncates integers and turns any non-zero into ``True``, so a seam averaging 200.5 comes
    back 200 and a mask seam averaging 0.5 comes back set.
    """
    if image.dtype == dtype:
        return image
    if dtype == torch.bool:  # a bool image is a mask, so cut it rather than rounding
        return image > 0.5
    if not dtype.is_floating_point:
        return image.round().clamp(torch.iinfo(dtype).min, torch.iinfo(dtype).max).to(dtype)
    return image.to(dtype)


def _resample(image: torch.Tensor, size: tuple) -> torch.Tensor:
    """Bilinear resize; torch resamples floats only, so anything else goes via float and back."""
    if image.is_floating_point():
        return F.interpolate(image, size=size, mode="bilinear", align_corners=False)
    out = F.interpolate(image.float(), size=size, mode="bilinear", align_corners=False)
    return restore_dtype(out, image.dtype)


@torch.inference_mode()
def upscale_image(image: torch.Tensor, size: tuple, mode: ScaleMode = ScaleMode.PADDING) -> torch.Tensor:
    """Upscale image to the desired size via either padding or interpolation.

    Args:
        image (torch.Tensor): Image (b,c,h,w)
        size (tuple): tuple to which image is upscaled.
        mode (str, optional): Upscaling mode. Defaults to "padding".

    Returns:
        Tensor: Upscaled image.
    """
    image_h, image_w = image.shape[2:]
    resize_h, resize_w = size

    if mode == ScaleMode.PADDING:
        pad_h = resize_h - image_h
        pad_w = resize_w - image_w
        image = F.pad(image, [0, pad_w, 0, pad_h])
    elif mode == ScaleMode.INTERPOLATION:
        image = _resample(image, (resize_h, resize_w))
    else:
        msg = f"Unknown mode {mode}. Only padding and interpolation is available."
        raise ValueError(msg)

    return image


@torch.inference_mode()
def downscale_image(image: torch.Tensor, size: tuple, mode: ScaleMode = ScaleMode.PADDING) -> torch.Tensor:
    """Opposite of upscaling. This image downscales image to a desired size.

    Args:
        image (torch.Tensor): Input image
        size (tuple): Size to which image is down scaled.
        mode (str, optional): Downscaling mode. Defaults to "padding".

    Returns:
        Tensor: Downscaled image
    """
    input_h, input_w = size
    if mode == ScaleMode.PADDING:
        image = image[:, :, :input_h, :input_w]
    elif mode == ScaleMode.INTERPOLATION:
        image = _resample(image, (input_h, input_w))  # bilinear, to match upscale_image
    else:
        msg = f"Unknown mode {mode}. Only padding and interpolation is available."
        raise ValueError(msg)

    return image


def as_int(x):
    if isinstance(x, torch.Tensor):
        return int(x.detach().cpu().item())
    return int(x)


class _OutGrid(NamedTuple):
    """The grid ``untile`` writes into. Equal to the image grid unless the tiles are scaled feature maps."""

    tile_size: list
    stride: list
    scale_size: list
    positions_h: list
    positions_w: list
    im_size: list


class Tiler:
    logger = logging.getLogger("Tiler")

    EXPECTED_FIELDS = {"tile_size", "stride", "im_size", "scale_size", "batch_size", "num_channel", "n_tiles"}

    def __init__(self, tile_size, stride, scale_mode="padding", overlap_mode="average"):
        """init tiler

        Args:
            tile_size (int | list): a int if tile_h equals to tile_w or a list of [tile_h, tile_w]
            stride (int | list): a int if stride_h equals to stride_w or a list of [stride_h, stride_w]
            scale_mode (str | ScaleMode): how tile() fits the image to the grid and untile() undoes it
            overlap_mode (str | OverlapMode): default blending of overlapping tiles in untile()
        """
        if isinstance(tile_size, int):
            tile_size = [tile_size] * 2
        if isinstance(stride, int):
            stride = [stride] * 2

        self.validate_tile_and_stride(tile_size, stride)

        self.tile_size = [as_int(v) for v in tile_size]  # 32.0 passes validation but slicing and F.pad need ints
        self.stride = [as_int(v) for v in stride]
        self.im_size: list = None
        self.scale_size: list = None
        self.batch_size: int = None
        self.num_channel: int = None
        self.n_tiles: list = None
        self._blend_mask_cache = {}  # Cache for blend masks by overlap mode
        self.scale_mode = ScaleMode(scale_mode)
        self.overlap_mode = OverlapMode(overlap_mode)

    @classmethod
    def validate_tile_and_stride(cls, tile_size, stride):
        """Validate that tile size and stride are set correctly."""
        if not tile_size or not stride:
            raise ValueError("Tile size and stride must be set.")
        if not isinstance(tile_size, list) or len(tile_size) != 2:
            raise ValueError(f"tile size must be a list of two elements. Got: {tile_size}")
        if not isinstance(stride, list) or len(stride) != 2:
            raise ValueError(f"stride must be a list of two elements. Got: {stride}")
        for name, values in (("tile size", tile_size), ("stride", stride)):
            for v in values:
                v = float(v.item()) if isinstance(v, torch.Tensor) else float(v)
                if v < 1 or v != int(v):
                    raise ValueError(f"{name} must be whole numbers of at least 1. Got: {values}")
        if stride[0] > tile_size[0] or stride[1] > tile_size[1]:
            raise ValueError("Stride size must be smaller or equal to tile size")

    @classmethod
    def from_json(cls, json_path):
        """init tiler from a json file

        Args:
            json_path (str): path to a metadata json
        """
        with open(json_path, "r") as file:
            return cls.from_dict(json.load(file))

    @classmethod
    def from_dict(cls, metadata: dict):
        """init tiler from a metadata dict

        Args:
            metadata (dict): metadata dictionary
        """
        if not metadata:
            raise ValueError("Metadata dictionary cannot be empty")

        tile_size = metadata.get("tile_size")
        stride = metadata.get("stride")
        if tile_size is None or stride is None:
            raise ValueError("Metadata dictionary must contain 'tile_size' and 'stride'")

        obj = cls(tile_size, stride)
        for k, v in metadata.items():
            if k in cls.EXPECTED_FIELDS and getattr(obj, k, None) is None:
                setattr(obj, k, v)
        if metadata.get("scale_mode") is not None:
            obj.scale_mode = ScaleMode(metadata["scale_mode"])
        return obj

    def to_dict(self):
        """save tiler metadata to a dict

        Returns:
            dict: tiler metadata
        """
        metadata = {}
        for field in self.EXPECTED_FIELDS:
            value = getattr(self, field, None)
            if value is None:
                raise RuntimeError(f"Tiler metadata incomplete. Missing field: {field}")
            metadata[field] = value
        # not in EXPECTED_FIELDS: metadata written before this existed must still load
        metadata["scale_mode"] = self.scale_mode.value
        return metadata

    def write_metadata(self, out_path):
        """write tiler metadata to a json file

        Args:
            out_path (str | Path): a output folder or a output file path
        """

        def save_json(data, json_file):
            with open(json_file, "w") as f:
                json.dump(data, f)

        out_path = Path(out_path)
        ext = out_path.suffix.lower()
        if ext == ".json":
            out_path.parent.mkdir(parents=True, exist_ok=True)
            save_json(self.to_dict(), out_path)
        else:
            out_path.mkdir(parents=True, exist_ok=True)
            save_json(self.to_dict(), out_path / "metadata.json")

    def _validate_state(self):
        """Validate that all required state is set"""
        missing = [f for f in self.EXPECTED_FIELDS if getattr(self, f, None) is None]
        if missing:
            raise RuntimeError(f"Tiler state incomplete. Missing: {missing}. Call tile() first or ensure metadata contains all fields.")

    @torch.inference_mode()
    def tile(self, im: torch.Tensor) -> torch.Tensor:
        """generate tiles from the image, scaled to the tile grid with ``self.scale_mode``.

        Args:
            im (Tensor): input image in the format: [b,c,h,w]

        Returns:
            Tensor: resized tiles
        """
        mode = self.scale_mode
        self.batch_size, self.num_channel, im_h, im_w = im.shape
        self.im_size = [im_h, im_w]
        device = im.device

        # scale image
        self.scale_size = list(compute_new_edges([im_h, im_w], self.tile_size, self.stride))
        resized_im = upscale_image(im, self.scale_size, mode)

        if self.scale_size[0] != im_h or self.scale_size[1] != im_w:
            if mode == ScaleMode.INTERPOLATION:
                self.logger.debug(f"resize img from {self.im_size} to {self.scale_size}")
            elif mode == ScaleMode.PADDING:
                self.logger.debug(f"pad img from {self.im_size} to {self.scale_size}")

        rows, cols = self._grid_positions()
        self.n_tiles = [len(rows), len(cols)]

        tiles = torch.zeros(
            (len(rows), len(cols), self.batch_size, self.num_channel, *self.tile_size),
            dtype=resized_im.dtype,
            device=device,
        )
        for (x, i), (y, j) in product(enumerate(rows), enumerate(cols)):
            tiles[x, y, :, :, :] = resized_im[:, :, i : i + self.tile_size[0], j : j + self.tile_size[1]]

        return tiles.contiguous().view(-1, self.num_channel, *self.tile_size)

    def _grid_positions(self) -> Tuple[list, list]:
        """Top-left rows and columns of the tile grid in the scaled image; ``product(rows, cols)`` is tile order.

        Every caller that walks the grid goes through here, so tile order, box order and untile order cannot drift.
        """
        tile_h, tile_w = as_int(self.tile_size[0]), as_int(self.tile_size[1])
        stride_h, stride_w = as_int(self.stride[0]), as_int(self.stride[1])
        rows = list(range(0, as_int(self.scale_size[0]) - tile_h + 1, stride_h))
        cols = list(range(0, as_int(self.scale_size[1]) - tile_w + 1, stride_w))
        return rows, cols

    def _checked_grid(self) -> Tuple[list, list]:
        """Grid positions, cross-checked against the recorded n_tiles, which restored metadata can contradict."""
        rows, cols = self._grid_positions()
        if [len(rows), len(cols)] != [as_int(n) for n in self.n_tiles]:
            raise ValueError(f"Metadata grid {self.n_tiles} does not match tile_size/stride/scale_size")
        return rows, cols

    def tile_boxes(self) -> torch.Tensor:
        """(n, 4) xyxy box per tile, one row per tile in the row-major order ``tile`` emits them.

        Coordinates are in the scaled image, so under padding the trailing row and column run past ``im_size``.
        """
        self._validate_state()
        tile_h, tile_w = as_int(self.tile_size[0]), as_int(self.tile_size[1])
        rows, cols = self._checked_grid()
        return torch.tensor([[j, i, j + tile_w, i + tile_h] for i, j in product(rows, cols)], dtype=torch.float32)

    @torch.inference_mode()
    def untile(
        self,
        tiles,
        overlap_mode=None,
        expected_scale=None,
    ):
        """convert tiles into original image, undoing ``self.scale_mode``. Apply blending for smooth transitions.

        Args:
            tiles (Torch): the tiles tensor in the format: [n_tiles*batch, c, tile_h, tile_w]
            overlap_mode (str | OverlapMode, optional): overlap handling mode. Defaults to self.overlap_mode.
            expected_scale (float, optional): require the tiles to be this multiple of ``tile_size``. Pass 1 for image
                tiles, so a wrong tile size raises instead of reconstructing at the wrong scale.

        Returns:
            Tensor: the reconstructed image with smooth blending
        """
        self._validate_state()
        overlap_mode = overlap_mode or self.overlap_mode
        if not isinstance(overlap_mode, (str, OverlapMode)):
            raise ValueError(f"overlap_mode must be str or OverlapMode enum. Got: {type(overlap_mode)}")

        # Convert string to enum if needed
        if not isinstance(overlap_mode, OverlapMode):
            overlap_mode = OverlapMode(overlap_mode)

        # anomalib traces this for export, and a traced shape is 0-dim tensors rather than ints
        n_tiles_total, num_channel, tile_h, tile_w = (as_int(d) for d in tiles.shape)
        batch_size = as_int(self.batch_size)
        expected_n_tiles = as_int(self.n_tiles[0]) * as_int(self.n_tiles[1]) * batch_size
        if n_tiles_total != expected_n_tiles:
            raise ValueError(f"Expected {expected_n_tiles} tiles, got {n_tiles_total}")

        grid = self._out_grid(tile_h, tile_w, expected_scale)
        positions = list(product(grid.positions_h, grid.positions_w))

        tiles = tiles.contiguous().view(-1, batch_size, num_channel, tile_h, tile_w)
        device = tiles.device

        # blending needs fractional weights, so the canvas is float whatever the tiles are
        work_dtype = torch.float64 if tiles.dtype == torch.float64 else torch.float32
        canvas = (batch_size, num_channel, *grid.scale_size)

        if overlap_mode == OverlapMode.MAX:
            # a zeroed canvas would floor the result, so start below every value a tile can hold
            im = torch.full(canvas, float("-inf"), dtype=work_dtype, device=device)
            for tile, (i, j) in zip(tiles, positions):
                # Take maximum between existing values and new tile
                im[:, :, i : i + tile_h, j : j + tile_w] = torch.maximum(im[:, :, i : i + tile_h, j : j + tile_w], tile)
            im = torch.where(im == float("-inf"), torch.zeros_like(im), im)  # isneginf has no ONNX export
        else:
            im = torch.zeros(canvas, dtype=work_dtype, device=device)
            weight_sum = torch.zeros(canvas, dtype=work_dtype, device=device)

            for tile, (i, j) in zip(tiles, positions):
                blend_mask = self._blend_mask(overlap_mode, device, i, j, grid)
                blend_mask_broadcast = blend_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, num_channel, -1, -1)

                im[:, :, i : i + tile_h, j : j + tile_w] += tile * blend_mask_broadcast
                weight_sum[:, :, i : i + tile_h, j : j + tile_w] += blend_mask_broadcast

            # clamp rather than add: a covered pixel must divide by its exact weight, not weight + eps
            im = torch.div(im, weight_sum.clamp(min=1e-8))

        return restore_dtype(downscale_image(im, grid.im_size, self.scale_mode), tiles.dtype)

    def _out_grid(self, tile_h: int, tile_w: int, expected_scale=None) -> "_OutGrid":
        """The grid ``untile`` writes into, at the scale of the tiles handed back.

        Tiles may be model feature maps rather than image tiles, so every part of the grid is rebuilt at their scale.
        """
        tile_size = [as_int(s) for s in self.tile_size]
        if expected_scale is not None:
            want = [int(round(s * expected_scale)) for s in tile_size]
            if [tile_h, tile_w] != want:
                raise ValueError(f"Expected tile size {want} at scale {expected_scale}, got [{tile_h}, {tile_w}]")

        scale_h, scale_w = tile_h / tile_size[0], tile_w / tile_size[1]
        if scale_h > 1 or scale_w > 1:  # a feature map is never larger than the tile it came from
            raise ValueError(f"Tiles [{tile_h}, {tile_w}] are larger than tile_size {tile_size}; untile does not upscale")

        rows, cols = self._checked_grid()

        # scale each position off its own row/column, so a non-integral scale cannot accumulate rounding error
        positions_h = [int(round(r * scale_h)) for r in rows]
        positions_w = [int(round(c * scale_w)) for c in cols]

        scale_size = [positions_h[-1] + tile_h, positions_w[-1] + tile_w]
        # padding: floor drops a partly padded cell and stays inside the canvas; interpolation has no padding, so take the nearest
        to_int = floor if self.scale_mode == ScaleMode.PADDING else round
        im_size = [max(1, to_int(as_int(self.im_size[0]) * scale_h)), max(1, to_int(as_int(self.im_size[1]) * scale_w))]
        # the blend ramp is sized from tile - stride, so stride has to describe the scaled grid, not the image grid
        stride = [
            max(1, positions_h[-1] // (len(rows) - 1)) if len(rows) > 1 else tile_h,
            max(1, positions_w[-1] // (len(cols) - 1)) if len(cols) > 1 else tile_w,
        ]
        return _OutGrid([tile_h, tile_w], stride, scale_size, positions_h, positions_w, im_size)

    def _blend_mask(self, overlap_mode: OverlapMode, device, i: int, j: int, grid: "_OutGrid") -> torch.Tensor:
        """(tile_h, tile_w) blend mask for the tile whose top-left sits at (i, j) in the output grid.

        Cached per overlap mode, grid, device and which sides have a neighbouring tile, so a grid needs at most 9 masks.
        """
        neighbours = (
            i > 0,
            j > 0,
            i + grid.tile_size[0] < grid.scale_size[0],
            j + grid.tile_size[1] < grid.scale_size[1],
        )
        key = (
            overlap_mode.value,
            tuple(grid.tile_size),
            tuple(grid.stride),
            device.type,
            device.index if device.index is not None else -1,
            neighbours,
        )
        if key not in self._blend_mask_cache:
            self._blend_mask_cache[key] = create_blend_mask(grid.tile_size, grid.stride, overlap_mode, device, neighbours)
        return self._blend_mask_cache[key]
