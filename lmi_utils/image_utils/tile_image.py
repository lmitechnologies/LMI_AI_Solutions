import argparse
import numpy as np
from pathlib import Path
from PIL import Image
import logging
import json


def __tile_image(source: Path, dest: Path, tile_w, tile_h, stride_w=None, stride_h=None):
    with Image.open(source) as img:
        width, height = img.size
        
        if stride_w is None:
            stride_w = tile_w
        if stride_h is None:
            stride_h = tile_h
        
        # resize to multiple of tile's width/height
        sx,sy = (width-tile_w)/stride_w, (height-tile_h)/stride_h
        resized = img.resize((stride_w*round(sx)+tile_w,stride_h*round(sy)+tile_h))
        width, height = resized.size
        
        logging.warning(f'resized img size: {resized.size}')
        
        x_steps = round(sx) + 1
        y_steps = round(sy) + 1

        tiles = {}
        for x in range(0, x_steps * stride_w, stride_w):
            for y in range(0, y_steps * stride_h, stride_h):
                # Define the box to cut out
                box = (x, y, x + tile_w, y + tile_h)
                # Create the tile
                tile = resized.crop(box) 
                tiles[f'{round(y/stride_h)}-{round(x/stride_w)}'] = tile
        
        for key in tiles.keys():
            tiles[key].save(dest / Path(source.stem + '-' + key + '.png'))
        
        # write metadata for entire image reconstruction
        with open(dest / Path(source.stem + '.json'), 'w') as f:
            json.dump({
                'width': width,
                'height': height,
                'tile_width': tile_w,
                'tile_height': tile_h,
                'stride_width': stride_w,
                'stride_height': stride_h,
            }, f)


def tile_image(source, dest, tile_w, tile_h, type, stride_w=None, stride_h=None):
    src_path = Path(source)
    dest_path = Path(dest)
    
    if src_path.is_file():
        __tile_image(src_path, dest_path, tile_w, tile_h, stride_w, stride_h)
    elif src_path.is_dir():
        for file in src_path.glob(f'*.{type}'):
            __tile_image(file, dest_path, tile_w, tile_h, stride_w, stride_h)


def reconstruct_image(source, dest, overlap_mode):
    src_path = Path(source)
    dest_path = Path(dest)
    files = {}
    for file_path in src_path.glob('*.png'):
        l = file_path.stem.split('-')
        coord = l[-2:]
        key = '-'.join(l[:-2])
        
        if files.get(key) is None:
            files[key] = {
                int(coord[0]): {
                    int(coord[1]): file_path
                }
            }
        else:
            if files[key].get(int(coord[0])) is None:
                files[key][int(coord[0])] = {
                    int(coord[1]): file_path
                }
            else:
                files[key][int(coord[0])][int(coord[1])] = file_path 
                
    for file_name in files.keys():
        # load metadata
        with open(src_path / Path(file_name + '.json'), 'r') as f:
            metadata = json.load(f)
            width, height = metadata['width'], metadata['height']
            tile_w, tile_h = metadata['tile_width'], metadata['tile_height']
            stride_w, stride_h = metadata['stride_width'], metadata['stride_height']

        # init the output image with -1
        with Image.open(files[file_name][0][0]) as tile:
            tile = np.array(tile)
            if tile.ndim==2:
                new_image = np.ones((height,width))*-1
            else:
                new_image = np.ones((height,width,3))*-1

        rows = sorted(files[file_name].keys())
        for r in rows:
            cols = sorted(files[file_name][r].keys())
            for c in cols:
                print(f"{c}, {r}")
                try:
                    with Image.open(files[file_name][r][c]) as tile:
                        tile = np.array(tile)
                        x, y = c*stride_w, r*stride_h
                        overlap = new_image[y:y+tile_h,x:x+tile_w]!=-1
                        # assign values to no overlap area
                        new_image[y:y+tile_h,x:x+tile_w][~overlap] = tile[~overlap]
                        # deal with overlap area
                        if overlap_mode=='max':
                            new_image[y:y+tile_h,x:x+tile_w][overlap] = np.maximum(new_image[y:y+tile_h,x:x+tile_w][overlap],tile[overlap])
                        elif overlap_mode=='avg':
                            new_image[y:y+tile_h,x:x+tile_w][overlap] = 0.5*(new_image[y:y+tile_h,x:x+tile_w][overlap]+tile[overlap])
                        else:
                            raise Exception(f'unknown overlap mode: {overlap_mode}')
                except FileNotFoundError:
                    continue
                
        new_image = Image.fromarray(new_image.astype('uint8'))
        new_image.save(dest_path / Path(file_name + '.png'))


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument('--option', required=True)
    ap.add_argument('--src', required=True)
    ap.add_argument('--dest', required=True)
    ap.add_argument('--w', type=int, default=224, help='tile width')
    ap.add_argument('--h', type=int, default=224, help='tile height')
    ap.add_argument('--stride_w', type=int, default=None, help='stride width, default to tile width')
    ap.add_argument('--stride_h', type=int, default=None, help='stride height, default to tile height')
    ap.add_argument('--overlap_mode', default='max', help='the supported modes: max, avg')
    ap.add_argument('--type', default='png')

    args=vars(ap.parse_args())

    if args['option'] == 'tile':
        tile_image(args['src'], args['dest'], args['w'], args['h'], args['type'], args['stride_w'], args['stride_h'])
    elif args['option'] == 'reconstruct':
        reconstruct_image(args['src'], args['dest'], args['overlap_mode'])
