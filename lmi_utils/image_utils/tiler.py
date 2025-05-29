from enum import Enum
from itertools import product
from math import ceil
import os
import logging
import json
import torch
from torch.nn import functional as F


logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


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


def compute_new_edges(edges:list, tile_size:list, stride:list):
    def __compute_new_edge(edge, tile, stride):
        if (edge-tile) % stride != 0:
            return tile + max(0,ceil((edge-tile)/stride)*stride)
        return edge
    
    out_h = __compute_new_edge(edges[0],tile_size[0],stride[0])
    out_w = __compute_new_edge(edges[1],tile_size[1],stride[1])
    return out_h,out_w


def create_blend_mask(tile_size: list, stride: list, overlap_mode: OverlapMode = OverlapMode.AVERAGE, device='cpu') -> torch.Tensor:
    """Create a blending mask for tile transitions.
    
    Args:
        tile_size (list): [tile_h, tile_w]
        stride (list): [stride_h, stride_w]
        overlap_mode (OverlapMode): Type of blending to apply
        device: Device to create tensor on
        
    Returns:
        torch.Tensor: Blending mask of shape [tile_h, tile_w]
    """
    tile_h, tile_w = tile_size
    stride_h, stride_w = stride
    
    # Calculate overlap regions
    overlap_h = tile_h - stride_h
    overlap_w = tile_w - stride_w
    
    if overlap_h <= 0 or overlap_w <= 0:
        # no overlap
        return torch.ones(tile_h, tile_w, device=device)
    
    mask = torch.ones(tile_h, tile_w, device=device)
    
    if overlap_mode == OverlapMode.AVERAGE:
        return mask
    
    if overlap_mode == OverlapMode.MAX:
        return mask
    
    # distance-based blending
    y_coords = torch.arange(tile_h, device=device).float()
    x_coords = torch.arange(tile_w, device=device).float()
    
    # Calculate distance from edges
    y_dist_from_top = y_coords
    y_dist_from_bottom = tile_h - 1 - y_coords
    x_dist_from_left = x_coords
    x_dist_from_right = tile_w - 1 - x_coords
    
    # create 2D grids
    y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
    
    # calculate minimum distance to any edge
    y_edge_dist = torch.minimum(y_dist_from_top, y_dist_from_bottom)
    x_edge_dist = torch.minimum(x_dist_from_left, x_dist_from_right)
    
    # apply blending in overlap regions only
    blend_region_h = overlap_h // 2
    blend_region_w = overlap_w // 2
    
    if overlap_mode == OverlapMode.LINEAR:
        y_blend = torch.clamp(y_edge_dist / blend_region_h, 0, 1)
        x_blend = torch.clamp(x_edge_dist / blend_region_w, 0, 1)
        mask = torch.minimum(y_blend, x_blend)
        
    elif overlap_mode == OverlapMode.COSINE:
        y_blend = torch.clamp(y_edge_dist / blend_region_h, 0, 1)
        x_blend = torch.clamp(x_edge_dist / blend_region_w, 0, 1)
        y_blend = 0.5 * (1 + torch.cos(torch.pi * (1 - y_blend)))
        x_blend = 0.5 * (1 + torch.cos(torch.pi * (1 - x_blend)))
        mask = torch.minimum(y_blend, x_blend)
        
    elif overlap_mode == OverlapMode.GAUSSIAN:
        center_h, center_w = tile_h // 2, tile_w // 2
        y_dist_center = torch.abs(y_grid - center_h)
        x_dist_center = torch.abs(x_grid - center_w)
        
        sigma_h = blend_region_h / 2
        sigma_w = blend_region_w / 2
        
        gaussian_y = torch.exp(-(y_dist_center ** 2) / (2 * sigma_h ** 2))
        gaussian_x = torch.exp(-(x_dist_center ** 2) / (2 * sigma_w ** 2))

        mask = torch.minimum(gaussian_y, gaussian_x)
    
    return mask


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
        image = F.interpolate(input=image, size=(resize_h, resize_w))
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
        image = F.interpolate(input=image, size=(input_h, input_w))
    else:
        msg = f"Unknown mode {mode}. Only padding and interpolation is available."
        raise ValueError(msg)

    return image


class Tiler:
    logger = logging.getLogger('Tiler')
    
    def __init__(self, tile_size, stride):
        """init tiler

        Args:
            tile_size (int | list): a int if tile_h equals to tile_w or a list of [tile_h, tile_w]
            stride (int | list): a int if stride_h equals to stride_w or a list of [stride_h, stride_w]
        """
        if isinstance(tile_size, int):
            tile_size = [tile_size]*2
        if isinstance(stride, int):
            stride = [stride]*2
            
        if not isinstance(tile_size, list) or len(tile_size)!=2:
            raise Exception(f'tile size must be a list of two elements. Got: {tile_size}')
        if not isinstance(stride, list) or len(stride)!=2:
            raise Exception(f'stride must be a list of two elements. Got: {stride}')
        if stride[0]>tile_size[0] or stride[1]>tile_size[1]:
            raise Exception('Stride size must be smaller or equal to tile size')
        
        self.tile_size = tile_size
        self.stride = stride
        self.im_size: list
        self.scale_size: list
        self.batch_size: int
        self.num_channel: int
        self.n_tiles: list
        self._blend_mask_cache = {}  # Cache for blend masks by overlap mode
        
        
    @classmethod
    def from_json(cls, json_path):
        """init tiler from a json file

        Args:
            json_path (str): path to a metadata json
        """
        obj = cls(0,0) # init an obj using dummy sizes
        with open(json_path, 'r') as file:
            metadata = json.load(file)
            
        for k,v in metadata.items():
            setattr(obj,k,v)
        return obj
        
    
    @torch.inference_mode()
    def tile(self, im:torch.Tensor, mode=ScaleMode.PADDING) -> torch.Tensor:
        """generate tiles from the image. Will resize images if necessary.

        Args:
            im (Tensor): input image in the format: [b,c,h,w]
            mode (ScaleMode, optional): scale mode. Defaults to ScaleMode.PADDING.

        Returns:
            Tensor: resized tiles
        """
        if not isinstance(mode, ScaleMode):
            raise Exception('mode must be a ScaleMode object')
        self.batch_size,self.num_channel,im_h,im_w = im.shape
        self.im_size = [im_h,im_w]
        device = im.device
        
        # scale image
        self.scale_size = compute_new_edges([im_h,im_w],self.tile_size,self.stride)
        resized_im = upscale_image(im,self.scale_size,mode)
        
        if self.scale_size[0]!=im_h or self.scale_size[1]!=im_w:
            if mode==ScaleMode.INTERPOLATION:
                self.logger.warning(f'resize img from {self.im_size} to {self.scale_size}')
            elif mode==ScaleMode.PADDING:
                self.logger.warning(f'pad img from {self.im_size} to {self.scale_size}')
        
        n_tiles_h = int((self.scale_size[0]-self.tile_size[0])/self.stride[0]) + 1
        n_tiles_w = int((self.scale_size[1]-self.tile_size[1])/self.stride[1]) + 1
        self.n_tiles = [n_tiles_h,n_tiles_w]
        
        tiles = torch.zeros((n_tiles_h,n_tiles_w,self.batch_size,self.num_channel,*self.tile_size),dtype=resized_im.dtype,device=device)
        for i,j in product(range(0,self.scale_size[0]-self.tile_size[0]+1,self.stride[0]),
                           range(0,self.scale_size[1]-self.tile_size[1]+1,self.stride[1])):
            x,y = i//self.stride[0],j//self.stride[1]
            tiles[x,y,:,:,:] = resized_im[:,:,i:i+self.tile_size[0],j:j+self.tile_size[1]]
        
        return tiles.contiguous().view(-1,self.num_channel,*self.tile_size)
    
    
    @torch.inference_mode()
    def untile(self, tiles, scale_mode=ScaleMode.PADDING, overlap_mode: OverlapMode = OverlapMode.AVERAGE):
        """convert tiles into original image. Apply blending for smooth transitions.

        Args:
            tiles (Torch): the tiles tensor in the format: [n_tiles*batch, c, tile_h, tile_w]
            mode (ScaleMode, optional): scale mode. Defaults to ScaleMode.PADDING.
            overlap_mode (OverlapMode, optional): overlap handling mode. Defaults to OverlapMode.AVERAGE.

        Returns:
            Tensor: the reconstructed image with smooth blending
        """
        if not isinstance(scale_mode, ScaleMode):
            raise Exception('mode must be a ScaleMode object')
        if not isinstance(overlap_mode, OverlapMode):
            raise Exception('overlap_mode must be an OverlapMode object')
        
        _,num_channel,tile_h,tile_w = tiles.shape
        tiles = tiles.contiguous().view(-1,self.batch_size,num_channel,tile_h,tile_w)
        device = tiles.device
        
        im = torch.zeros(self.batch_size,num_channel,*self.scale_size,device=device)
        
        if overlap_mode == OverlapMode.MAX:
            for tile,(i,j) in zip(tiles, product(range(0,self.scale_size[0]-self.tile_size[0]+1,self.stride[0]),
                                          range(0,self.scale_size[1]-self.tile_size[1]+1,self.stride[1]))):
                
                # Take maximum between existing values and new tile
                im[:,:,i:i+self.tile_size[0],j:j+self.tile_size[1]] = torch.maximum(
                    im[:,:,i:i+self.tile_size[0],j:j+self.tile_size[1]], 
                    tile
                )
        else:
            cache_key = (overlap_mode.value, str(device))
            if cache_key not in self._blend_mask_cache:
                self._blend_mask_cache[cache_key] = create_blend_mask(
                    self.tile_size, self.stride, overlap_mode, device
                )
            
            blend_mask = self._blend_mask_cache[cache_key]
            weight_sum = torch.zeros(self.batch_size,num_channel,*self.scale_size,device=device)
            
            blend_mask_broadcast = blend_mask.unsqueeze(0).unsqueeze(0).expand(
                self.batch_size, num_channel, -1, -1
            )
            
            for tile,(i,j) in zip(tiles, product(range(0,self.scale_size[0]-self.tile_size[0]+1,self.stride[0]),
                                          range(0,self.scale_size[1]-self.tile_size[1]+1,self.stride[1]))):
                
                weighted_tile = tile * blend_mask_broadcast
                
                im[:,:,i:i+self.tile_size[0],j:j+self.tile_size[1]] += weighted_tile
                weight_sum[:,:,i:i+self.tile_size[0],j:j+self.tile_size[1]] += blend_mask_broadcast
            
            eps = 1e-8
            im = torch.div(im, weight_sum + eps)
        
        return downscale_image(im,self.im_size,scale_mode).to(tiles.dtype)
    
    
    def write_metadata(self, out_path):
        """write tiler metadata to a json file

        Args:
            out_path (str): a output folder or a output file path
        """
        def save_json(data, json_file):
            with open(json_file, 'w') as f:
                json.dump(data,f)
        
        ext = os.path.splitext(out_path)[-1]
        if ext=='.json':
            os.makedirs(os.path.dirname(out_path),exist_ok=True)
            save_json(self.__dict__,out_path)
        else:
            os.makedirs(out_path,exist_ok=True)
            save_json(self.__dict__,os.path.join(out_path,'metadata.json'))