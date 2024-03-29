#built-in packages
import os
import glob
import shutil
import logging
import json

#3rd party packages
import cv2

#LMI packages
from label_utils.bbox_utils import rescale_oriented_bbox, convert_ls_obb_to_yolo, get_lst_bbox_to_xywh
from image_utils.pad_image import fit_image_to_size


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)



def pad_with_json(path_json, image_dir,output_dir, output_imsize, keep_same_filename=True):
    with open(path_json) as f:    
        l = json.load(f)
    for dt in l:
        # load file name
        fname = dt['data']['image'].split("=")[-1]
        fname = os.path.basename(fname)
        input_path = os.path.join(image_dir,fname)
        
        # pad the image
        fit_image_to_size(input_path, output_dir, output_imsize, keep_same_filename=keep_same_filename)
        
        # load annotations
        for annot in dt['annotations']:
            num_bbox = len(annot['result'])
            if num_bbox>0:
                logger.info(f'{num_bbox} annotation(s) in {fname}')
            for result in annot['result']:
                # rescale the bbox
                bbox = result['value']
                pixel_x, pixel_y, pixel_width, pixel_height = get_lst_bbox_to_xywh(bbox['x'],bbox['y'], bbox['width'],bbox['height'],result['original_width'], result['original_height'])
                result["value"]['x'] = pixel_x
                result["value"]['y'] = pixel_y
                result["value"]['width'] = pixel_width
                result["value"]['height'] = pixel_height
                result['original_width'] = output_imsize[0]
                result['original_height'] = output_imsize[1]
                result['value']['yolo_obb'] = convert_ls_obb_to_yolo(result)# added yolo obb bbox into the json
    return l
                    

if __name__=='__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--path_imgs', '-i', required=True, help='the path to images')
    ap.add_argument('--path_json', required=True, help='[optinal] the path of a json file that corresponds to path_imgs')
    ap.add_argument('--out_imsz', required=True, help='the output image size [w,h], w and h are separated by a comma')
    ap.add_argument('--path_out', '-o', required=True, help='the path to resized images')
    
    args = vars(ap.parse_args())

    output_imsize = list(map(int,args['out_imsz'].split(',')))
    assert len(output_imsize)==2, 'the output image size must be two ints'
    logger.info(f'output image size: {output_imsize}')
    
    path_imgs = args['path_imgs']
    path_out = args['path_out']
    path_json = os.path.join(path_imgs, args['path_json'])
    
    #check if annotation exists
    if not os.path.isfile(path_json):
        raise Exception(f'cannot find file: {path_json}')
    
    # create output path
    assert path_imgs != path_out, 'input and output path must be different'
    if not os.path.isdir(path_out):
        os.makedirs(path_out)

    #resize images with annotation csv file
    updated_json = pad_with_json(path_json, path_imgs,path_out,output_imsize)
    with open(os.path.join(path_out, "labels.json"), 'w') as f:
        json.dump(updated_json, f, indent=4)
    logger.info(f'updated json file saved to: {path_json}')
    
    