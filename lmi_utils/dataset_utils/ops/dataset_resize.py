import logging
import numpy as np

#LMI packages
from image_utils.img_resize import resize


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def resize_annotations(shapes, orig_h: int, orig_w: int, new_h: int, new_w: int):
    """resize shapes in-place

    Args:
        shapes (Shape): a list of Shape objects
        rx (float): resize ratio in x direction
        ry (float): resize ratio in y direction
    """
    for annot in shapes:
        annot.value = annot.value.resize(orig_h, orig_w, new_h, new_w)
    
    return shapes

def resize_dataset(dataset, images, output_imsize, maintain_aspect_ratio=False):
    """
    resize images and its annotations with a csv file
    if the aspect ratio changes, it will generate warnings.
    Arguments:
        path_imgs(str): the image folder
        path_json(str): the path of csv annotation file
        output_imsize(list): a list of output image size [w,h]
    Return:
        shapes(dict): the map <original image name, a list of shape objects>, where shape objects are annotations
    """
    resized_images = {}
    for f in dataset.files:
        file_path = f.path
        im = images[file_path]
        h,w = im.shape[:2]
        f.height = h
        f.width = w
        
        # resize image
        tw,th = output_imsize
        
        
        if tw is None and th is None:
            # raise Exception('Both width and height cannot be None')
            tw,th = w,h
            im2 = im
        else:
            if maintain_aspect_ratio and (tw is not None or th is not None):
                scale = max(tw,th) / max(w,h)
                tw = np.int32(scale * w)
                th = np.int32(scale * h)
                im2 = resize(im, width=tw, height=th)
                
            else:    
                if tw is None:
                    tw = 'w'
                    im2 = resize(im, height=th)
                elif th is None:
                    th = 'h'
                    im2 = resize(im, width=tw)
                else:
                    im2 = resize(im, width=tw, height=th)
            
        resized_images[file_path] = im2

        th,tw = im2.shape[:2]
        if tw != w or th != h:
            shapes = resize_annotations(f.annotations,orig_h=h, orig_w=w, new_h=tw, new_w=th)
            f.annotations = shapes
        logger.info(f'resize {file_path} from w:{w} h:{h} to w:{tw} h:{th}')
        f.height = th
        f.width = tw
    return resized_images, dataset
