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


class OverlapMode(str, Enum): # New enum for overlap handling
    """Type of mode for handling overlapping tiles.""" # New enum for overlap handling
    AVERAGE = "average" # New enum for overlap handling
    MAX = "max" # New enum for overlap handling


def compute_new_edges(edges:list, tile_size:list, stride:list):
    def __compute_new_edge(edge, tile, stride):
        if (edge-tile) % stride != 0:
            return tile + max(0,ceil((edge-tile)/stride)*stride)
        return edge

    out_h = __compute_new_edge(edges[0],tile_size[0],stride[0])
    out_w = __compute_new_edge(edges[1],tile_size[1],stride[1])
    return out_h,out_w


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
    def untile(self, tiles, scale_mode=ScaleMode.PADDING, overlap_mode=OverlapMode.AVERAGE): 
        """convert tiles into original image. Handles overlapping tiles by averaging or taking the max.

        Args:
            tiles (Torch): the tiles tensor in the format: [n_tiles*batch, c, tile_h, tile_w]
            scale_mode (ScaleMode, optional): scale mode for final image. Defaults to ScaleMode.PADDING.
            overlap_mode (OverlapMode, optional): how to handle overlapping regions. Defaults to OverlapMode.AVERAGE.

        Returns:
            Tensor: the reconstructed image
        """
        if not isinstance(scale_mode, ScaleMode): 
            raise Exception('scale_mode must be a ScaleMode object') 
        if not isinstance(overlap_mode, OverlapMode): 
            raise Exception('overlap_mode must be an OverlapMode object') 

        _,num_channel,tile_h,tile_w = tiles.shape
        tiles_reshaped = tiles.contiguous().view(-1,self.batch_size,num_channel,tile_h,tile_w) 
        device = tiles_reshaped.device

        im = torch.zeros(self.batch_size,num_channel,*self.scale_size,device=device, dtype=tiles_reshaped.dtype) 
        if overlap_mode == OverlapMode.AVERAGE: 
            cnts = torch.zeros(self.batch_size,num_channel,*self.scale_size,device=device, dtype=torch.float32) 
            ones = torch.ones(self.batch_size,num_channel,*self.tile_size,device=device, dtype=torch.float32) 

        processed_tiles = 0 
        for idx, (i, j) in enumerate(product(range(0, self.scale_size[0] - self.tile_size[0] + 1, self.stride[0]),
                                             range(0, self.scale_size[1] - self.tile_size[1] + 1, self.stride[1]))):
            current_tile = tiles_reshaped[idx] 
            if overlap_mode == OverlapMode.AVERAGE:
                im[:,:,i:i+self.tile_size[0],j:j+self.tile_size[1]] += current_tile
                cnts[:,:,i:i+self.tile_size[0],j:j+self.tile_size[1]] += ones
            elif overlap_mode == OverlapMode.MAX: 
                
                if processed_tiles == 0 and i == 0 and j == 0:
                     im[:,:,i:i+self.tile_size[0],j:j+self.tile_size[1]] = current_tile
                else:
                    im_slice = im[:,:,i:i+self.tile_size[0],j:j+self.tile_size[1]] 
                    im[:,:,i:i+self.tile_size[0],j:j+self.tile_size[1]] = torch.maximum(im_slice, current_tile) # Take the maximum
            processed_tiles +=1


        if overlap_mode == OverlapMode.AVERAGE: # Perform division only for average mode
            # average the overlapping tiles
            im = torch.div(im,cnts.clamp(min=1)) # Clamped cnts to avoid division by zero, ensure it's float
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