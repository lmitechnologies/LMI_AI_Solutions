import cv2
import argparse
import os
import argparse
import logging

#LMI packages
from dataset_utils.representations import Dataset
from dataset_utils.ops.dataset_resize import resize_dataset
from dataset_utils.ops.dataset_pad import pad_dataset
from dataset_utils.ops.dataset_rotate import rotate_dataset
from dataset_utils.ops.dataset_crop_by_label import crop_dataset_by_label


logger = logging.getLogger(__name__)

def load_images(path_imgs,dataset):
    """
    load images from path_imgs
    """
    images = {}
    for f in dataset.files:
        file_path = f.path
    
        p = os.path.join(path_imgs, file_path)
        if not os.path.isfile(p):
            raise Exception(f'cannot find file: {file_path}')
        im = cv2.imread(p)
        if im is None:
            raise Exception(f'cannot read image: {file_path}')
        images[file_path] = im
    return images

def generate_image_name(image_name, args):
    if args['operation'] == 'resize':
        out_name = os.path.splitext(image_name)[0] + f"_{args['operation']}_{args['width']}x{args['height']}" + '.png'
    elif args['operation'] == 'pad':
        out_name = os.path.splitext(image_name)[0] + f"_{args['operation']}_{args['width']}x{args['height']}" + '.png'
    elif args['operation'] == 'rotate':
        out_name = os.path.splitext(image_name)[0] + f"_{args['operation']}_{-1*args['angle']}" + '.png' # -1 so that the angle is positive for clockwise rotation
    elif args['operation'] == 'crop-by-label':
        out_name = os.path.splitext(image_name)[0] + f"_{args['operation']}_{args['target_label']}" + '.png'
    else:
        out_name = os.path.splitext(image_name)[0] + f"_{args['operation']}" + '.png'
    
    return out_name

def save_dataset(dataset, images,path_out_images, out_json, args):
    for f in dataset.files:
        file_path = f.path
        im_out = images[file_path]
        im_name = os.path.basename(file_path)
        out_h, out_w = im_out.shape[:2]
        if f'id{f.id}_' not in im_name:
            im_name = f'id{f.id}_{im_name}'
        out_name = generate_image_name(im_name, args)
        absolute_image_path = os.path.join(path_out_images, out_name)
        relative_image_path = os.path.relpath(absolute_image_path, path_out_images)
        cv2.imwrite(os.path.join(path_out_images,out_name), im_out)
        f.update_file(path=relative_image_path, width=out_w, height=out_h, id=f.id)
    if not out_json.endswith('.json') and out_json!='labels.json':
        if not os.path.isdir(out_json):
            os.makedirs(out_json)
        out_json = os.path.join(out_json, 'labels.json')
    elif out_json == 'labels.json':
        out_json = os.path.join(path_out_images, 'labels.json')
    else:
        out_json = out_json
    dataset.save(out_json)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Process images with operations: resize, pad, or rotate."
    )
    
    # Common arguments for all operations
    parser.add_argument(
        '--path_imgs', '-i', required=True,
        help='Path to the input images.'
    )
    parser.add_argument(
        '--path_json', default='labels.json',
        help='Optional JSON file corresponding to the images (default: "labels.json").'
    )
    parser.add_argument(
        '--path_out_images', '-oi', required=True,
        help='Path to save output images.'
    )
    parser.add_argument(
        '--path_out_json', '-of', default='labels.json',
        help='Path to store the output JSON file (default: "labels.json").'
    )
    parser.add_argument(
        '--bg', action='store_true',
        help='Save background images that have no labels.'
    )
    parser.add_argument(
        '--warn_crop', action='store_true',
        help='If enabled, will warn if labels are cropped.'
    )
    
    # Create subparsers for each operation
    subparsers = parser.add_subparsers(dest='operation', required=True,
                                       help='Image operation to perform (resize, pad, rotate, crop).')
    
    # Create a parent parser for commands that require dimensions
    dim_parser = argparse.ArgumentParser(add_help=False)
    dim_parser.add_argument(
        '--width', type=int, required=False, default=None,
        help='Output image width.'
    )
    dim_parser.add_argument(
        '--height',type=int, required=False, default=None,
        help='Output image height.'
    )
    
    # resize, pad, and rotate parsers
    resize_parser = subparsers.add_parser('resize', parents=[dim_parser],
                                          help='Resize images')
    resize_parser.add_argument(
        '--par', '-par', action='store_true',
        help='Maintain aspect ratio when resizing and pad when needed.'
    )
    pad_parser = subparsers.add_parser('pad', parents=[dim_parser],
                                       help='Pad images')
    rotate_parser = subparsers.add_parser('rotate', help='Rotate images')
    rotate_parser.add_argument(
        '--angle', type=float, required=False, default=90,
        help='Angle (in degrees) to rotate images.'
    )
    
    rotate_parser.add_argument('--counter-clockwise', action='store_true', help='rotate the images counter-clockwise', default=False, required=False)
    
    # crop by label parser
    crop_by_label = subparsers.add_parser('crop-by-label', help='Crop images by label')
    
    crop_by_label.add_argument(
        '--target_label', type=str, required=True,
        help='Bbox label to crop images and labels.'
    )
    
    return vars(parser.parse_args())


def apply_ops(args):
    if args.get('width', None) == 0:
        args['width'] = None
    if args.get('height', None) == 0:
        args['height'] = None
    
    if args['operation'] in ['resize', 'pad']:
        output_imsize = [args['width'], args['height']]
    else:
        output_imsize = None
    logger.debug(f'output image size: {output_imsize}')
    
    path_imgs = args['path_imgs']
    path_out = args['path_out_images']
    path_json = args['path_json'] if os.path.isfile(args['path_json']) else os.path.join(path_imgs, args['path_json'])
    out_json = args['path_out_json']
    
    # check if annotation exists
    if not os.path.isfile(path_json):
        raise Exception(f'cannot find file: {path_json}. Please create an empty json file, if there are no labels.')
    
    # create output path
    assert path_imgs!=path_out, 'input and output path must be different'
    if not os.path.isdir(path_out):
        os.makedirs(path_out)
    
    # determine warning level for annotation crop
    crop_warning_level = logging.WARNING if args['warn_crop'] else logging.DEBUG

    # load dataset
    dataset = Dataset.load(path_json)
    images = load_images(path_imgs, dataset)
    output_images = images
    output_dataset = dataset
    # perform operation
    
    # Resize images
    if args['operation'] == 'resize':
        output_images, output_dataset = resize_dataset(dataset, images, output_imsize, args['par'])
        if args['par']:
            if args['width'] is not None and args['height'] is not None:
                output_images, output_dataset = pad_dataset(output_dataset, output_images, output_imsize)
    
    # Pad images
    elif args['operation'] == 'pad':
        logger.debug(f'Padding images to size: {output_imsize}')
        output_images, output_dataset = pad_dataset(dataset, images, output_imsize, crop_warning_level=crop_warning_level)
    
    # Rotate images
    elif args['operation'] == 'rotate':
        logger.debug(f'Rotating images by {args["angle"]} degrees')
        output_images, output_dataset = rotate_dataset(dataset, images, args['angle'], args['counter_clockwise'])
    
    # Crop images by label
    elif args['operation'] == 'crop-by-label':
        logger.debug(f'Cropping images by label: {args["target_label"]}')
        output_images, output_dataset = crop_dataset_by_label(dataset, images, args['target_label'])
    
    if not args['bg']:
        # remove files with no annotations
        output_dataset.delete_empty_files()
        
    # save images and dataset
    save_dataset(output_dataset, output_images, path_out_images=path_out,out_json=out_json, args=args)
    logger.debug(f'output json file: {out_json}')
    

def main():
    args = parse_args()
    apply_ops(args)


if __name__=='__main__':
    main()

        
    