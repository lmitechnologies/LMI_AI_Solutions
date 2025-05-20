import logging
import numpy as np
import cv2

#LMI packages
from dataset_utils.representations import Annotation
from image_utils.img_resize import resize


logger = logging.getLogger(__name__)

def resize_annotations(shapes, orig_h: int, orig_w: int, new_h: int, new_w: int):
    """resize shapes

    Args:
        shapes (Shape): a list of Shape objects
        rx (float): resize ratio in x direction
        ry (float): resize ratio in y direction
    """
    shapes = shapes.copy()
    for annot in shapes:
        annot.value = annot.value.resize(orig_h, orig_w, new_h, new_w)

    return shapes

def resize_annotated_image(image: np.ndarray, annotations: list[Annotation], width: int, height: int, maintain_aspect_ratio=False) -> tuple[np.ndarray, list[Annotation]]:
    """
    resize image and its annotations to the size [width, height]
    Arguments:
        image(np.ndarray): the input image
        annotations(list): a list of annotation objects
        width(int): the target width of the output image
        height(int): the target height of the output image
        maintain_aspect_ratio(bool): whether to maintain the aspect ratio of the image
    Return:
        im_out(np.ndarray): the resized image
        annotations(list): a list of annotation objects
    """
    h, w = image.shape[:2]
    th, tw  = height, width
    
    if (tw is None and th is None) or (tw == w and th == h):
        th, tw = h, w
        im_out = image
    else:
        if maintain_aspect_ratio:
            scale = min(th / h, tw / w)
            tw = np.int32(scale * w)
            th = np.int32(scale * h)
            im_out = resize(image, width=tw, height=th)            
        else:    
            if tw is None:
                tw = w
                im_out = resize(image, height=th)
            elif th is None:
                th = h
                im_out = resize(image, width=tw)
            else:
                im_out = resize(image, width=tw, height=th)
        
    th,tw = im_out.shape[:2]
    if tw != w or th != h:
        annotations = resize_annotations(annotations,orig_h=h, orig_w=w, new_h=th, new_w=tw)

    return im_out, annotations    


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
      
        im_out, annot_out = resize_annotated_image(image=im, annotations=f.annotations, width=output_imsize[0], height=output_imsize[1], maintain_aspect_ratio=maintain_aspect_ratio)

        logger.debug(f'resize {file_path} from w:{im.shape[1]} h:{im.shape[0]} to w:{im_out.shape[1]} h:{im_out.shape[0]}')

        f.height =  im_out.shape[0]
        f.width = im_out.shape[1]
        f.annotations = annot_out
        resized_images[file_path] = im_out
    return resized_images, dataset
