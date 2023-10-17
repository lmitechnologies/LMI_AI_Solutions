#%%
import cv2
import glob
import numpy as np
import os
from image_utils.rgb_converter import convert_array_to_rainbow
import matplotlib.pyplot as plt
import time

BLACK=[0,0,0]

def img_rgb_to_int_array(img):
    if len(img.shape)>2:
        arr=img.reshape(-1,3)
        arr_int=[]
        for element in arr:
            r=element[0]
            g=element[1]
            b=element[2]
            binary_word = (r << 16) | (g << 8) | b
            arr_int.append(binary_word)
        arr_int=np.array(arr_int)
    else:
        arr_int=img
    return arr_int

def blend(img_p, img_i):
    # Blend the images
    alpha = 0.3  # Weight for image1
    beta = 0.7  # Weight for image2
    gamma = 0  # Scalar added to each pixel
    blended_image = cv2.addWeighted(img_p, alpha, img_i, beta, gamma)
    return blended_image

# Process data
def preprocess_INT16(img,map_choice='rainbow-med'):

    TWO_TO_24_MINUS_ONE=np.power(2,24)-1
    levels=np.unique(img)
    # Fetch first valid z height
    level_1=levels[1].astype(np.float32)
    # Fetch indices of invalid height values
    empty_ind=np.where(img==0)
    img=img.astype(np.float32)
    # Shift imag 1 unit above the floor to maximize the full scale range
    img=img-(level_1-1)
    # Reset the floor to 0
    img[empty_ind]=0

    # Normalize image
    img_max=img.max()
    img_n=img/img_max

    # Convert to Grayscale
    if map_choice=='gray':   
        hmap=(img_n*255.0).astype(np.uint8)

    # Convert to Rainbow Low Precision
    elif map_choice=='rainbow-low':
        img_gray=(img_n*255.0).astype(np.uint8)
        hmap=cv2.applyColorMap(img_gray,cv2.COLORMAP_JET)
        hmap=cv2.cvtColor(hmap,cv2.COLOR_BGR2RGB)
        hmap[empty_ind]=BLACK

    # %% Convert to Med precision Rainbow
    elif map_choice=='rainbow-med':
        img_fsr=(img_n*TWO_TO_24_MINUS_ONE).astype(np.uint32)
        hmap=convert_array_to_rainbow(img_fsr)

    elif map_choice=='rainbow-high':
        #TODO: This doesn't seem to improve quantization
        colormap = plt.get_cmap('jet')
        img_rgba = (colormap(img_n)*255.0).astype(np.uint8)
        # hmap=colormap(img_n)
        hmap=cv2.cvtColor(img_rgba,cv2.COLOR_RGBA2RGB)
        hmap[empty_ind]=BLACK

    else:
        raise Exception(f'Unsupported colormapping option: {map_choice}')

    
    return hmap

if __name__=='__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('-i','--input_path', default='./')
    ap.add_argument('-o','--output_path', default=None)
    mapping_options = ["gray", "rainbow-low", "rainbow-med","rainbow-high"]
    ap.add_argument('--map_choice', choices=mapping_options,help="Mapping choices: gray, rainbow-low, rainbow-med, rainbow-high")
    ap.add_argument('--show_quant', action='store_true', help="Show quantization results (SLOW!).")
    filtering_options=[None]
    
    args = vars(ap.parse_args())

    inpath=args['input_path']
    outpath=args['output_path']
    if outpath is None:
        outpath=inpath
    map_choice=args['map_choice']
    if map_choice is None:
        map_choice='rainbow-med'

    if os.path.isdir(inpath):
        files=glob.glob(os.path.join(inpath,'*.png'))
    else:
        files=[inpath]
    
    if not os.path.exists(outpath):
        os.path.makedirs(outpath)
    proc_time=[]
    for file in files:
        img_p=cv2.imread(file,-1)
        unique_input_value=len(np.unique(img_p))
        print(f'[INFO] Input image has {unique_input_value} unique values.')
        t0=time.time()
        hmap=preprocess_INT16(img_p,map_choice=map_choice)
        t1=time.time()
        img_bgr=cv2.cvtColor(hmap,cv2.COLOR_RGB2BGR)
        tdelta=t1-t0
        print(f'[INFO] Proc time = {tdelta}')
        if args['show_quant']:
            hmap_int=img_rgb_to_int_array(hmap)
            unique_input_value=len(np.unique(hmap_int))
            print(f'[INFO] Converted hmap has {unique_input_value} unique values.')
        proc_time.append(tdelta)
        fname=os.path.split(file)[1]
        fname=fname.replace('.png','_hmap.png')
        cv2.imwrite(os.path.join(outpath,fname),img_bgr)
        # plt.imsave(os.path.join(outpath,fname), img_rainbow)

    proc_time=np.array(proc_time)
    print(f'[INFO] Mean Processing Time = {proc_time.mean()}')
        



