import cv2
import numpy as np
from os import listdir, makedirs
from os.path import isfile, join, isdir

class NumpyUtils():
    def png_to_npy(self,source_path, destination_path, rotate=False, rgb2bgr=False):
        files = [f for f in listdir(source_path) if isfile(join(source_path, f)) and ( (".png" in f) or (".jpg" in f) )]

        for f in files:
            print(join(source_path, f))
            np_frame = cv2.imread(join(source_path, f))
            np_frame=cv2.cvtColor(np_frame,cv2.COLOR_RGB2BGR)
            if rotate:
                np_frame = np.rot90(np_frame)
            if rgb2bgr:
                np_frame=cv2.cvtColor(np_frame,cv2.COLOR_RGB2BGR)
            np.save(join(destination_path, f.replace('.png', '.npy')), np_frame)

    def npy_to_png(self,source_path, destination_path, rotate=False, rgb2bgr=False):
        files = [f for f in listdir(source_path) if isfile(join(source_path, f)) and ".npy" in f]

        for f in files:
            print(join(source_path, f))
            np_frame = np.load(join(source_path, f))
            if rotate:
                np_frame = np.rot90(np_frame)
            if rgb2bgr:
                np_frame=cv2.cvtColor(np_frame,cv2.COLOR_RGB2BGR)
            np_frame=cv2.cvtColor(np_frame,cv2.COLOR_RGB2BGR)
            cv2.imwrite(join(destination_path, f.replace('.npy', '.png')), np_frame)
    
    def png_to_png(self,source_path, destination_path, rotate=False, rgb2bgr=False):
        files = [f for f in listdir(source_path) if isfile(join(source_path, f)) and ".png" in f]

        for f in files:
            print(join(source_path, f))
            np_frame = cv2.imread(join(source_path, f))
            if rotate:
                np_frame = np.rot90(np_frame)
            if rgb2bgr:
                np_frame=cv2.cvtColor(np_frame,cv2.COLOR_RGB2BGR)
            np_frame=cv2.cvtColor(np_frame,cv2.COLOR_RGB2BGR)
            cv2.imwrite(join(destination_path, f), np_frame)

if __name__=="__main__":
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument('--option',required=True,help='npy_2_png or png_2_npy or png_2_png')
    ap.add_argument('--src',required=True)
    ap.add_argument('--dest',required=True)
    ap.add_argument('--rotate', action='store_true',help='rotate image to 90 degree')
    ap.add_argument('--rgb2bgr',action='store_true',help='apply rgb to bgr correction')
    
    args=vars(ap.parse_args())
    option=args['option']
    src=args['src']
    dest=args['dest']
    rotate = args['rotate']
    rgb2bgr=args['rgb2bgr']

    translate=NumpyUtils()

    print(f'Src: {src}')
    print(f'Dest: {dest}')
    
    if not isdir(dest):
        makedirs(dest)

    if option=='npy_2_png':
        translate.npy_to_png(src,dest,rotate,rgb2bgr)
    elif option=='png_2_npy':
        translate.png_to_npy(src,dest,rotate,rgb2bgr)
    elif option=='png_2_png':
        translate.png_to_png(src,dest,rotate,rgb2bgr)
    else:
        raise Exception('Input option must be npy_2_png, png_2_npy, or png_2_png')
    
    
    
