import os
import subprocess
import cv2
import random
import tarfile
import logging
import numpy as np

logging.getLogger().setLevel(logging.INFO)

CAMERAS = ['avt','gocator'] 


def load_zst_img(file_path):
    newpath = file_path.replace('.zst','')
    cmds = ['unzstd', '-f', f'{file_path}', '-o', newpath]
    subprocess.run(' '.join(cmds), shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    img = cv2.imread(newpath, cv2.IMREAD_UNCHANGED)
    # logging.info(f'load type: {img.dtype} with min and max: {img.min()}, {img.max()}')
    return img


def extract_imgs(input_path, out_path, target_cam='all', num_imgs=20, first_dir='first_dir',last_dir='last_dir', task='anomdet' ,random_list=False, seed=777):
    sku_path = os.path.expanduser(input_path)
    out_path = os.path.expanduser(out_path)

    idx1 = os.path.basename(sku_path).find('archive-')
    idx2 = os.path.basename(sku_path).find('_')
    if idx1==-1 or idx2==-1:
        raise Exception('failed to match sku folder naming convention')
    sku = os.path.basename(sku_path)[idx1+8:idx2]
    logging.info(f'sku: {sku}')

    out_path = os.path.join(out_path, sku, 'data')
    for camera in CAMERAS:
        tar_cam,tar_cam_id = target_cam.split('_')
        logging.info(f'target camera: "{tar_cam}", target sensor id: "{tar_cam_id}"')
        tar_cam = tar_cam.lower()
        if tar_cam!='all' and camera != tar_cam:
            logging.warning(f'found camera "{camera}" mismatch with target camera "{tar_cam}", skip')
            continue
        tar_path=os.path.join(sku_path, 'sensor', 'gadget-sensor-'+camera)
        for sensor_id in os.listdir(tar_path):
            sensor_path = os.path.join(tar_path, sensor_id)
            if not os.path.isdir(sensor_path):
                logging.warning(f'cannot found the sensor path: {sensor_path}, skip')
                continue
            if tar_cam_id!='' and tar_cam_id!=sensor_id:
                logging.warning(f'sensor_id of "{sensor_id}" mismatch with target_id of "{tar_cam_id}", skip')
                continue
            logging.info(sensor_path)
            
            cnt = 0
            output_path = os.path.join(out_path, camera+'_'+sensor_id, 'label-'+task)
            files=os.listdir(sensor_path)
            if random_list:
                random.seed(seed)
                random.shuffle(files)
            logging.info(f'working on {len(files)} images...')
            
            for filename in files:
                file_path = os.path.join(sensor_path,filename)
                split_folder = first_dir if cnt<num_imgs else last_dir
                
                if os.path.isdir(file_path):
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
                    # create subfolder
                    subfolder = os.path.join(sensor_path, name)
                    os.makedirs(subfolder, exist_ok=True)
                    
                    # unzip the tar file
                    with tarfile.open(file_path) as f:
                        f.extractall(subfolder)
                    
                    # save intensity and profile image
                    for tt in ['intensity','profile']:
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
                            logging.exception(f'cannot find {tt} img after untar')
                        final_out_path=os.path.join(output_path, tt, split_folder)
                        os.makedirs(final_out_path, exist_ok=True)
                        cv2.imwrite(os.path.join(final_out_path, name+'.'+tt+'.png'), img)
                    cnt += 1
                elif ext=='.tiff':
                    # skip tiff
                    pass
                else:
                    logging.warning(f'cannot recognize the type: {ext}')



if __name__=='__main__':
    import argparse
    ap = argparse.ArgumentParser(description='this script extracts raw sensor images from gofactory archived download data')
    ap.add_argument('--input_path', '-i', required=True, help='the input sku path. It must include the SKU name, such as: 2023-02-23/archive-WT10-HR1001-04929_63f7d4ac47411d08ebf31daa')
    ap.add_argument('--out_path', '-o', required=True, help='the output path')
    ap.add_argument('--num_imgs', '-n', default=20, type=int, help='the number of images kept for training, the rest will be put in untracked-data subfolder')
    ap.add_argument('--target_camera', '-t', default='all', help='the target sensor for parsing data. The format could be either "avt_1" or "avt".')
    ap.add_argument('--first_dir','-f',default='label-objdet')
    ap.add_argument('--last_dir','-l',default='untracked-data')
    ap.add_argument('--random',action='store_true',help='Randomize the list')
    ap.add_argument('--task',default='anomdet',nargs='?',choices=['anomdet','objdet'],help='create the task subfolder inside the sensor path')
    ap.add_argument('--seed',default=777,type=int,help='the random seed')
    args = ap.parse_args()
    
    extract_imgs(args.input_path, args.out_path, args.target_camera, args.num_imgs, args.first_dir, args.last_dir, args.task, args.random, args.seed)
    