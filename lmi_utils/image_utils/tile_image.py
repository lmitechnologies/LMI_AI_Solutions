import argparse
import numpy as np
from pathlib import Path
from PIL import Image

def __tile_image(source: Path, dest: Path, tile_w, tile_h):
    with Image.open(source) as img:
        width, height = img.size
        
        # resize to multiple of tile's width/height
        resized = img.resize(((round(width/tile_w)*tile_w),(round(height/tile_h)*tile_h)))
        
        width, height = resized.size
        x_tiles = width // tile_w
        y_tiles = height // tile_h

        tiles = {}
        for x in range(0, x_tiles * tile_w, tile_w):
            for y in range(0, y_tiles * tile_h, tile_h):
                # Define the box to cut out
                box = (x, y, x + tile_w, y + tile_h)
                # Create the tile
                tile = resized.crop(box) 
                tiles[f'{round(y/tile_h)}{round(x/tile_w)}'] = tile
        
        for key in tiles.keys():
            tiles[key].save(dest / Path(source.stem + '-' + key + '.png'))

def tile_image(source, dest, tile_w, tile_h, type):
    src_path = Path(source)
    dest_path = Path(dest)
    
    if src_path.is_file():
        __tile_image(src_path, dest_path, tile_w, tile_h)
    elif src_path.is_dir():
        for file in src_path.glob(f'*.{type}'):
            __tile_image(file, dest_path, tile_w, tile_h)


def reconstruct_image(source, dest):
    src_path = Path(source)
    dest_path = Path(dest)
    files = {}
    for file_path in src_path.glob('*.png'):
        key, coord = file_path.stem.split('-')
        
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
        grid_w = len(files[file_name][0].keys())
        grid_h = len(files[file_name].keys())

        width, height = Image.open(files[file_name][0][0]).size

        new_image = Image.new('RGB', ((grid_w*width), (grid_h*height)))

        rows = sorted(files[file_name].keys())
        for r in rows:
            cols = sorted(files[file_name][r].keys())
            for c in cols:
                print(f"{c}, {r}")
                try:
                    with Image.open(files[file_name][r][c]) as tile:
                        new_image.paste(tile, (c*width, r*height))
                except FileNotFoundError:
                    continue
    
        new_image.save(dest_path / Path(file_name + '.png'))


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument('--option', required=True)
    ap.add_argument('--src', required=True)
    ap.add_argument('--dest', required=True)
    ap.add_argument('--w', type=int, default=244)
    ap.add_argument('--h', type=int, default=244)
    ap.add_argument('--type', default='png')

    args=vars(ap.parse_args())

    if args['option'] == 'tile':
        tile_image(args['src'], args['dest'], args['h'], args['w'], args['type'])
    elif args['option'] == 'reconstruct':
        reconstruct_image(args['src'], args['dest'])
