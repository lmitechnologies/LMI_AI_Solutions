import glob
import random
import os
import random
import shutil
import cv2

def get_files(dir):
    ftypes=('*.png','*.jpg','*.bmp')
    files_grabbed = []
    for ftype in ftypes:
        files_grabbed.extend(glob.glob(os.path.join(dir,ftype)))
    return files_grabbed

def randomize(files_list,seed=42):
    random.seed(seed)
    random.shuffle(files_list)
    return files_list

def split_files(files_list,n):
    training=files_list[0:n]
    test=[]
    if len(files_list)>n:
        test=files_list[n:]
    return training,test

def move_files(dir,training,test,convert_to_png,rotate_png_90):
    training_path=os.path.join(dir,'training/')
    test_path=os.path.join(dir,'test/')
    if os.path.exists(training_path):
        shutil.rmtree(training_path)
    os.makedirs(training_path)
    if os.path.exists(test_path):
        shutil.rmtree(test_path)
    os.makedirs(test_path)

    for file in training:
        fname=os.path.split(file)[1]
        if convert_to_png:
            print(f'[INFO] Converting {fname} to .png and moving to training directory')
            img=cv2.imread(file)
            if rotate_png_90:
                img=cv2.rotate(img,cv2.ROATE_90_CLOCKWISE)
            ext=os.path.splitext(fname)[1]
            fname_out=fname.replace(ext,'.png')
            path_out=os.path.join(training_path,fname_out)
            cv2.imwrite(path_out,img)
            print(f'[INFO] Removing original {fname} to .png')
            os.remove(file)
        else:
            print(f'[INFO] moving {fname} to training directory.')
            path_out=os.path.join(training_path,fname)
            shutil.move(file,path_out)
    for file in test:
        fname=os.path.split(file)[1]
        if convert_to_png:
            print(f'[INFO] Converting {fname} to .png and moving to test dirctory')
            img=cv2.imread(file)
            ext=os.path.splitext(fname)[1]
            fname_out=fname.replace(ext,'.png')
            path_out=os.path.join(test_path,fname_out)
            cv2.imwrite(path_out,img)
            print(f'[INFO] Removing original {fname} to .png')
            os.remove(file)
        else:
            print(f'[INFO] moving {fname} to test directory.')
            path_out=os.path.join(test_path,fname)
            shutil.move(file,path_out)

if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help='Source data directory.')
    parser.add_argument("--training_size",type=int,default=0, help="Training data size.")
    parser.add_argument("--make_random", action='store_true')
    parser.add_argument("--convert_to_png", action='store_true')
    parser.add_argument("--rotate_png_90",action='store_true')
    args = parser.parse_args()

    files_list=get_files(args.data_dir)
    if args.make_random:
        files_list=randomize(files_list)
    training_list,test_list=split_files(files_list,args.training_size)

    move_files(args.data_dir,training_list,test_list,args.convert_to_png,args.rotate_png_90)





