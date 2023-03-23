from PIL import Image
import pickle
import numpy
from os import listdir, makedirs
from os.path import isfile, join, isdir

class GadgetImageUtils():

    SCHEMA_ID: str = "gadget2d" 
    VERSION: int = 1

    def pkl_2_npy(self, source_path, destination_path, rotate=False):
        files = [f for f in listdir(source_path) if isfile(join(source_path, f)) and ".gadget2d.pickle" in f]

        for file in files:
            print(join(source_path, file))
            
            with open(join(source_path, file), "rb") as f:
                content = pickle.load(f)
            npy_arr = content["pixel_array"]

            if rotate:
                npy_arr = numpy.rot90(npy_arr)

            numpy.save(join(destination_path, file.replace('.gadget2d.pickle', '.npy')), npy_arr)

    def pkl_2_png(self, source_path, destination_path, rotate=False):
        files = [f for f in listdir(source_path) if isfile(join(source_path, f)) and ".gadget2d.pickle" in f]

        for file in files:
            print(join(source_path, file))
            
            with open(join(source_path, file), "rb") as f:
                content = pickle.load(f)
            
            npy_arr = content["pixel_array"]

            if rotate:
                npy_arr = numpy.rot90(npy_arr)

            image = Image.fromarray(npy_arr)
            image.save(join(destination_path, file.replace('.gadget2d.pickle', '.png')))
    
    def npy_2_pkl(self, source_path, destination_path, rotate=False):
        files = [f for f in listdir(source_path) if isfile(join(source_path, f)) and ".npy" in f]

        for file in files:
            print(join(source_path, file))

            npy_arr = numpy.load(join(source_path, file))

            if npy_arr.ndim == 3 and npy_arr.shape[2] == 3:
                pixel_format = "RGB_8"
            elif npy_arr.ndim == 2:
                pixel_format = "GRAY_8"
            else:
                raise ValueError("Unsupported array shape.")

            if rotate:
                npy_arr = numpy.rot90(npy_arr)

            content = { 
                "metadata": {
                    "schema": self.SCHEMA_ID,
                    "version": self.VERSION, 
                    "pixel_format": pixel_format
                }, 
                "pixel_array": npy_arr,
            }
            
            with open(join(destination_path, file.replace('.npy', '.gadget2d.pickle')), "wb") as f:
                pickle.dump(content, f, protocol=4)

    def png_2_pkl(self, source_path, destination_path, rotate=False):
        files = [f for f in listdir(source_path) if isfile(join(source_path, f)) and ".png" in f]

        for file in files:
            print(join(source_path, file))

            img = Image.open(join(source_path, file))
            npy_arr = numpy.array(img)

            if npy_arr.ndim == 3 and npy_arr.shape[2] == 3:
                pixel_format = "RGB_8"
            elif npy_arr.ndim == 2:
                pixel_format = "GRAY_8"
            else:
                raise ValueError("Unsupported array shape.")

            if rotate:
                npy_arr = numpy.rot90(npy_arr)

            content = { 
                "metadata": {
                    "schema": self.SCHEMA_ID,
                    "version": self.VERSION, 
                    "pixel_format": pixel_format
                }, 
                "pixel_array": npy_arr,
            }
            
            with open(join(destination_path, file.replace('.png', '.gadget2d.pickle')), "wb") as f:
                pickle.dump(content, f, protocol=4)


if __name__=="__main__":
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument('--option',required=True,help='pkl_2_npy or pkl_2_png or npy_2_pkl or png_2_pkl')
    ap.add_argument('--src',required=True)
    ap.add_argument('--dest',required=True)
    ap.add_argument('--rotate', action='store_true',help='rotate image to 90 degree')
    
    args=vars(ap.parse_args())
    option=args['option']
    src=args['src']
    dest=args['dest']
    rotate = args['rotate']

    print(f"Rotate: {rotate}")
    translate=GadgetImageUtils()

    print(f'Src: {src}')
    print(f'Dest: {dest}')
    
    if not isdir(dest):
        makedirs(dest)

    if option=='pkl_2_npy':
        translate.pkl_2_npy(src,dest,rotate)
    elif option=='pkl_2_png':
        translate.pkl_2_png(src,dest,rotate)
    elif option=='npy_2_pkl':
        translate.npy_2_pkl(src,dest,rotate)
    elif option=='png_2_pkl':
        translate.png_2_pkl(src,dest,rotate)
    else:
        raise Exception('Input option must be pkl_2_npy, pkl_2_png, npy_2_pkl, or png_2_pkl')