import cv2
import numpy as np
from os import listdir
from os.path import isfile, join

class NumpyUtils():
    def png_to_npy(self,source_path, destination_path):
        files = [f for f in listdir(source_path) if isfile(join(source_path, f)) and ".png" in f]

        for f in files:
            print(join(source_path, f))
            np_frame = cv2.imread(join(source_path, f))
            np.save(join(destination_path, f.replace('.png', '.npy')), np_frame)

    def npy_to_png(self,source_path, destination_path):
        files = [f for f in listdir(source_path) if isfile(join(source_path, f)) and ".npy" in f]

        for f in files:
             print(join(source_path, f))
             np_frame = np.load(join(source_path, f))
             cv2.imwrite(join(destination_path, f.replace('.npy', '.png')), np_frame)

if __name__=="__main__":
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument('--option',required=True,help='npy_2_png or png_2_npy')
    ap.add_argument('--src',required=True)
    ap.add_argument('--dest',required=True)
    args=vars(ap.parse_args())
    option=args['option']
    src=args['src']
    dest=args['dest']

    translate=NumpyUtils()

    print(f'Src: {src}')
    print(f'Dest: {dest}')

    if option=='npy_2_png':
        translate.npy_to_png(src,dest)
    elif option=='png_2_npy':
        translate.png_to_npy(src,dest)
    else:
        raise Exception('Input option must be npy_2_png or png_2_npy')
    
    
    
