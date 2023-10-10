import os
import subprocess
import cv2
import random
import tarfile
import logging
import numpy as np

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

CAMERAS = ['avt','gocator'] 
ARCHIVE_PREFIX = 'archive-'
UNZIP_DIR = 'un_zipped'


def load_zst_img(file_path):
    newpath = file_path.replace('.zst','')
    cmds = ['unzstd', '-f', f'{file_path}', '-o', newpath]
    subprocess.run(' '.join(cmds), shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    img = cv2.imread(newpath, cv2.IMREAD_UNCHANGED)
    # logger.info(f'load type: {img.dtype} with min and max: {img.min()}, {img.max()}')
    return img


def unzip_tarfile(file_path, output_path):
    # unzip the tar file. Create output_path if it does not exist.
    if not os.path.isdir(output_path):
        os.makedirs(output_path)
    with tarfile.open(file_path) as f:
        f.extractall(output_path)


def extract_imgs(input_path, out_path, target_cam='all', num_imgs=20, first_dir='first_dir',last_dir='last_dir', task='anomdet', random_list=False, seed=777):
    sku_path = os.path.expanduser(input_path)
    out_path = os.path.expanduser(out_path)
    _,ext = os.path.splitext(sku_path)

    # get sku
    input_fname = os.path.basename(sku_path)
    idx1 = input_fname.find(ARCHIVE_PREFIX)
    if idx1 != -1:
        # found sku
        idx2 = input_fname.rfind('_')
        sku = input_fname[idx1+len(ARCHIVE_PREFIX):idx2]
    else:
        # find time stamp instead
        idx1 = input_fname.rfind('_')
        idx2 = input_fname.find('.') if ext == '.tar' else len(input_fname)
        sku = input_fname[idx1+1:idx2]
    if idx1==-1 or idx2==-1:
        raise Exception('archive folder naming convention is not expected. Support two formats: PATH/archive-SKU_TIMESTAMP or PATH/archive_TIMESTAMP')
    
    logger.info(f'sku path: {sku_path}')
    logger.info(f'sku: {sku}')
    
    # unzip if it's a tar file
    if ext == '.tar':
        unzip_folder = os.path.join(os.path.dirname(sku_path), input_fname[:-4])
        logger.info(f'found that the input path is a tar file. unzip it to {unzip_folder}')
        unzip_tarfile(sku_path, unzip_folder)
        sku_path = unzip_folder

    out_path = os.path.join(out_path, sku)
    for camera in CAMERAS:
        # get target camera name and ID
        l_c = target_cam.split('_')
        if len(l_c)==2:
            tar_cam,tar_cam_id = l_c
        elif len(l_c)==1:
            tar_cam = l_c[0]
            tar_cam_id = ''
        else:
            raise Exception(f'The target camera argument is not correct. Should be either "avt_1", "avt" or "all"')
        logger.info(f'target camera: "{tar_cam}", target sensor id: "{tar_cam_id}"')
        tar_cam = tar_cam.lower()
        if tar_cam!='all' and camera != tar_cam:
            logger.warning(f'found camera "{camera}" mismatch with target camera "{tar_cam}", skip')
            continue
        tar_path=os.path.join(sku_path, 'sensor', 'gadget-sensor-'+camera)
        if not os.path.isdir(tar_path):
            logger.warning(f'Not found sensor: {camera}, skip')
            continue
        for sensor_id in os.listdir(tar_path):
            sensor_path = os.path.join(tar_path, sensor_id)
            if not os.path.isdir(sensor_path):
                logger.warning(f'cannot found the sensor path: {sensor_path}, skip')
                continue
            if tar_cam_id!='' and tar_cam_id!=sensor_id:
                logger.warning(f'sensor_id of "{sensor_id}" mismatch with target_id of "{tar_cam_id}", skip')
                continue
            logger.info(sensor_path)
            
            cnt = 0
            output_path = os.path.join(out_path, camera+'_'+sensor_id, 'label-'+task)
            files=[f for f in os.listdir(sensor_path) if f!=UNZIP_DIR]
            if random_list:
                random.seed(seed)
                random.shuffle(files)
            logger.info(f'working on {len(files)} images...')
            
            for filename in files:
                file_path = os.path.join(sensor_path,filename)
                split_folder = first_dir if cnt<num_imgs else last_dir
                
                if os.path.isdir(file_path):
                    logger.info(f'found folder: {file_path}, skip')
                    continue
                
                name,ext = os.path.splitext(filename)
                ext = ext.lower()
                if ext=='.zst':
                    # decompress file and save as png
                    img = load_zst_img(file_path)
                    final_out_path = os.path.join(output_path, split_folder)
                    os.makedirs(final_out_path, exist_ok=True)
                    cv2.imwrite(os.path.join(final_out_path, name.replace('.tiff','.png')), img)
                    cnt += 1
                elif ext in ['.jpg','.jpeg']:
                    # load and save as png
                    img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
                    final_out_path = os.path.join(output_path, split_folder)
                    os.makedirs(final_out_path, exist_ok=True)
                    cv2.imwrite(os.path.join(final_out_path, name+'.png'), img)
                    cnt += 1
                elif ext=='.tar':
                    # create subfolder and unzip
                    subfolder = os.path.join(sensor_path, UNZIP_DIR , name)
                    unzip_tarfile(file_path, subfolder)
                    
                    # save intensity and profile image
                    for tt in ['intensity','profile']:
                        write_file=True
                        if os.path.isfile(os.path.join(subfolder, tt+'.tiff.zst')):
                            img_path = os.path.join(subfolder, tt+'.tiff.zst')
                            img = load_zst_img(img_path)
                        elif os.path.isfile(os.path.join(subfolder, tt+'.jpg')):
                            img_path = os.path.join(subfolder, tt+'.jpg')
                            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                        elif os.path.isfile(os.path.join(subfolder, tt+'.png')):
                            img_path = os.path.join(subfolder, tt+'.png')
                            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                        else:
                            logger.warning(f'cannot find {tt} img after untar')
                            write_file=False
                        if write_file: 
                            final_out_path=os.path.join(output_path, tt, split_folder)
                            os.makedirs(final_out_path, exist_ok=True)
                            cv2.imwrite(os.path.join(final_out_path, name+'.'+tt+'.png'), img)
                    cnt += 1
                elif ext=='.tiff':
                    # skip tiff
                    pass
                else:
                    logger.warning(f'cannot recognize the type: {ext}, skip file: {filename}')



if __name__=='__main__':
    import argparse
    ap = argparse.ArgumentParser(description='this script extracts raw sensor images from gofactory archive data')
    ap.add_argument('--input_path', '-i', required=True, help='the input sku path. Could be a single SKU tarfile/untar folder or a folder containing multiple SKUs')
    ap.add_argument('--out_path', '-o', required=True, help='the output path')
    ap.add_argument('--task',required=True,nargs=1,choices=['anomdet','objdet'],help='choose the task: "anomdet" or "objdet"')
    ap.add_argument('--num_imgs', '-n', default=20, type=int, help='the number of images kept for training. defaults to 20')
    ap.add_argument('--target_camera', '-t', default='all', help='the target sensor(s) for parsing data. The format could be either "avt_1" or "avt". default to "all"')
    ap.add_argument('--first_dir','-f',default='training-data',help='the folder name of training images. default to "training-data"')
    ap.add_argument('--last_dir','-l',default='untracked-data',help='the folder name of images that are NOT used for training. default to "untracked-data"')
    ap.add_argument('--random',action='store_true',help='randomly select training images')
    ap.add_argument('--seed',default=777,type=int,help='the random seed')
    args = ap.parse_args()
    
    if os.path.isfile(args.input_path):
        extract_imgs(args.input_path, args.out_path, args.target_camera, args.num_imgs, args.first_dir, args.last_dir, args.task[0], args.random, args.seed)
    
    if os.path.isdir(args.input_path):
        subfs = os.listdir(args.input_path)
        # single untar archive folder
        if args.input_path.find('archive') != -1 and 'sensor' in subfs:
            extract_imgs(args.input_path, args.out_path, args.target_camera, args.num_imgs, args.first_dir, args.last_dir, args.task[0], args.random, args.seed)
            exit(0)
        
        visited = set()
        for f in subfs:
            if f.find('archive') != -1:
                # incase parse the same SKU twice because there might exist the tarfile and the untar folder with same SKU
                key = f
                idx = f.find('.tar')
                if idx != -1:
                    key = f[:idx]
                if key not in visited:
                    extract_imgs(os.path.join(args.input_path, f), args.out_path, args.target_camera, args.num_imgs, args.first_dir, args.last_dir, args.task[0], args.random, args.seed)
                    visited.add(key)
                else:
                    logger.info(f'Found parsed data: {f}, skip')
            else:
                logger.info(f'Not find keyword "archive": {f}, skip')
        
