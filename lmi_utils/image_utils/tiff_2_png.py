import argparse
import glob
import os
import cv2

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument('-i','--input_path',default='.')
    ap.add_argument('-o','--output_path',default='./png')
    args=vars(ap.parse_args())
    input_path=args['input_path']
    output_path=args['output_path']
    wh_stretch=args['wh_stretch']

    if os.path.isdir(input_path):
        files=glob.glob(os.path.join(input_path,'*.tiff'))
    elif os.path.splitext[1]=='tiff':
        files=[input_path]
    else:
        raise Exception('Unrecognized input path.  Expecting directory or .png file.')

    if os.path.exists(output_path):
        pass
    else:
        os.mkdir(output_path)

    for file in files:
        img=cv2.imread(file)
        outfile=os.path.split(file)[1]
        outfile=os.path.splitext(outfile)[0]+'.png'
        outfile=os.path.join(output_path,outfile)
        cv2.imwrite(outfile,img)


