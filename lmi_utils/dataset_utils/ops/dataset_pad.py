import cv2
import numpy as np
import logging

#LMI packages
from dataset_utils.representations import AnnotationType, Box, Mask, Polygon, Point2d, Annotation
from gadget_utils.pipeline_utils import fit_array_to_size


logger = logging.getLogger(__name__)


def fit_shapes_to_size(shapes, pad_l, pad_t, pad_h, pad_w,orig_h,orig_w):
    """
    description:
        add the left and top paddings to the shapes
    arguments:
        shapes(list): a list of Shape objects (Rect or Mask)
        pad_l(int): the left paddings
        pad_t(int): the top paddings 
    """
    shapes = shapes.copy()
    for annot in shapes:
        annot.value = annot.value.pad(pad_h=pad_h, pad_w=pad_w, pl=pad_l, pt=pad_t, h=orig_h, w=orig_w)
    return shapes 


def pad_annotated_image(image: np.ndarray, annotations: list[Annotation], width: int, height: int) -> tuple[np.ndarray, list[Annotation], bool]:
    """
    description:
        pad the image to the size [width,height] and modify its annotations accordingly
    arguments:
        image(np.ndarray): the input image
        annotations(list): a list of annotation objects
        width(int): the target width of the output image
        height(int): the target height of the output image
    return:
        im_out(np.ndarray): the padded image
        annotations(list): a list of annotation objects
        is_warning(bool): whether the annotations have been clipped or removed
    """
    h, w = image.shape[:2]
    pw = width
    ph = height
    
    if ph is None and pw is None:
        ph = h
        pw = w

    # pad image
    im_out, pad_l, _, pad_t, _ = fit_array_to_size(image, pw, ph)
    pw = im_out.shape[1]
    ph = im_out.shape[0]
    
    # pad shapes
    annotations = fit_shapes_to_size(annotations,pad_l,pad_t, pad_h=ph, pad_w=pw, orig_h=h, orig_w=w)

    delete_ids,is_warning = clip_shapes(annotations, W=pw, H=ph)
    annotations = [shape for shape in annotations if shape.id not in delete_ids]

    return im_out, annotations, is_warning
   

def pad_dataset(dataset, images, output_imsize, crop_warning_level=logging.DEBUG):
    """
    pad/crop the image to the size [W,H] and modify its annotations accordingly
    arguments:
        input_path(str): the input image path
        json_path(str): the path to the json annotation file
        output_imsize(list): the width and height of the output image
    """
    
    padded_images = {}
    cnt_warnings = 0
    for f in dataset.files:        
        file_path = f.path
        
        im_out, annot_out, is_warning = pad_annotated_image(image=images[file_path], annotations=f.annotations, width=output_imsize[0], height=output_imsize[1])

        if is_warning:
            cnt_warnings += 1

        f.height =  im_out.shape[0]
        f.width = im_out.shape[1]
        f.annotations = annot_out
        padded_images[file_path] = im_out
    if cnt_warnings:
        logger.log(crop_warning_level, f'Labels were cropped for {cnt_warnings} images.')
    return padded_images, dataset


def clip_shapes(shapes, W, H, crop_warning_level=logging.DEBUG):
    """
    description:
        clip the shapes so that they are fit in the target size [W,H]
    """
    is_warning = False
    shapes = np.array(shapes)
    delete_ids = []
    for i,shape in enumerate(shapes):
        is_del = 0
        if shape.type == AnnotationType.BOX:
            
            box = shape.value.to_numpy()[:-1]
            new_box = np.clip(box, a_min=0, a_max=[W,H,W,H])
            
            if np.all(new_box==0) or new_box[0]==new_box[2] or new_box[1]==new_box[3]:
                is_del = 1
                delete_ids.append(shape.id)
                logger.log(crop_warning_level, f'Bounding box {box} was excluded by image dimensions [{W},{H}]')
            elif (np.any(new_box==W) and np.all(box!=W)) or (np.any(new_box==H) and np.all(box!=H)) \
                    or (np.any(new_box==0) and np.all(box!=0)):
                logger.log(crop_warning_level, f'Bounding box {box} was clipped to image dimensions [{W}, {H}]')
                is_warning = True
                
                shape.value = Box(*new_box)
        
        elif shape.type == AnnotationType.MASK or shape.type == AnnotationType.POLYGON:
            X,Y = shape.value.coords(w=W,h=H)
            new_X = np.clip(X, a_min=0, a_max=W)
            new_Y = np.clip(Y, a_min=0, a_max=H)
            if np.all(new_X==W) or np.all(new_Y==H) or np.all(new_X==0) or np.all(new_Y==0):
                is_del = 1
                delete_ids.append(shape.id)
                # logger.warning(f'polygon {[(x,y) for x,y in zip(new_X,new_Y)]} is outside of the size [{W},{H}]')
                logger.log(crop_warning_level, f'Polygon {shape.id} was excluded by image dimensions [{W},{H}]')

            elif (np.any(new_X==W) and np.all(X!=W)) or (np.any(new_Y==H) and np.all(Y!=H)) \
                or (np.any(new_X==0) and np.all(X!=0)) or (np.any(new_Y==0) and np.all(Y!=0)):
                logger.log(crop_warning_level, f'Polygon {shape.id} was clipped to image dimensions [{W}, {H}]')
                is_warning = True
                if shape.type == AnnotationType.POLYGON:
                    shape.value = Polygon(points=np.array(list(zip(new_X,new_Y))).astype(int).tolist())
                else:
                    img = np.zeros((H,W),dtype=np.uint8)
                    cv2.fillPoly(img, [np.array(list(zip(new_X,new_Y))).astype(int)], 1)
                    shape.value = Mask(mask=img)
                
        elif shape.type == AnnotationType.KEYPOINT:
            x,y = shape.value.coords()
            if x<0 or x>W or y<0 or y>H:
                is_del = 1
                logger.log(crop_warning_level, f'keypoint ({x},{y}) was excluded by image dimensions [{W},{H}]')
                delete_ids.append(shape.id)
            else:
                shape.value = Point2d(x=x, y=y)
                
        if is_del:
            is_warning = True
            
    return delete_ids, is_warning