import argparse
import glob
import os
import cv2


def stretch(img,wh_stretch):  
    h,w=img.shape[:2]
    if w>h:
        new_w=int(w/wh_stretch)
        print(f'[INFO] Changing w={w} to w={new_w}.')
        dim=(new_w,h)
        img_stretch=cv2.resize(img,dim,cv2.INTER_AREA)
    else:
        new_h=int(h*wh_stretch)
        print(f'[INFO] Changing h={h} to h={new_h}.')
        dim=(w,new_h)
        img_stretch=cv2.resize(img,dim,cv2.INTER_AREA)
    return img_stretch

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument('-i','--input_path',default='.')
    ap.add_argument('-o','--output_path',default='./stretch')
    ap.add_argument('--wh_stretch',type=float,default=3)
    args=vars(ap.parse_args())
    input_path=args['input_path']
    output_path=args['output_path']
    wh_stretch=args['wh_stretch']

    if os.path.isdir(input_path):
        files=glob.glob(os.path.join(input_path,'*.png'))
    elif os.path.splitext[1]=='png':
        files=[input_path]
    else:
        raise Exception('Unrecognized input path.  Expecting directory or .png file.')

    if os.path.exists(output_path):
        pass
    else:
        os.mkdir(output_path)

    for file in files:
        img=cv2.imread(file,-1)
        img_stretch=stretch(img,wh_stretch)
        outfile=os.path.split(file)[1]
        outfile=os.path.splitext(outfile)[0]+'_stretch.png'
        outfile=os.path.join(output_path,outfile)
        cv2.imwrite(outfile,img_stretch)


