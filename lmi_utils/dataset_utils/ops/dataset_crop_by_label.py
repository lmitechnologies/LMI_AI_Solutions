
import numpy as np
import logging

#LMI packages
from dataset_utils.representations import Box, Mask, Polygon, BoxAnnotation, MaskAnnotation, PolygonAnnotation

logger = logging.getLogger(__name__)

def crop_by_percent(image, crop_percent, crop_from = 'top'):
    height, width = image.shape[:2]
    x1, y1 = 0, 0
    x2, y2 = width, height
    if crop_from == 'top':
        x1, y1 = 0, 0
        x2, y2 = width, int(height * crop_percent)
    elif crop_from == 'bottom':
        x1, y1 = 0, int(height * (1 - crop_percent))
        x2, y2 = width, height
    return x1, y1, x2, y2

def crop_kp(bbox, shape):
    x1,y1,x2,y2 = bbox
    w,h = x2-x1, y2-y1    
    x,y = shape.x, shape.y
    x -= x1
    y -= y1
    
    valid = True
    if x<0 or x>=w or y<0 or y>=h:
        valid = False
        logger.warning(f'in {shape.im_name}, keypoint {x:.4f},{y:.4f} is out of the label bbox: {x1:.4f},{y1:.4f},{x2:.4f},{y2:.4f}. skip')
    
    return x,y,valid

def crop_bbox(bbox1, bbox2):
    crop_x1, crop_y1, _, _ = bbox1
    target_x1, target_y1, target_x2, target_y2 = bbox2
    adjusted_x1 = target_x1 - crop_x1
    adjusted_y1 = target_y1 - crop_y1
    adjusted_x2 = target_x2 - crop_x1
    adjusted_y2 = target_y2 - crop_y1
    return adjusted_x1, adjusted_y1, adjusted_x2, adjusted_y2

def crop_mask(bbox, mask=None, polygon_mask=None, bbox_format="xywh"):
    # Interpret the bounding box coordinates
    if bbox_format == "xywh":
        x, y, w, h = bbox
        xmin, ymin, xmax, ymax = x, y, x + w, y + h
    elif bbox_format == "xyxy":
        xmin, ymin, xmax, ymax = bbox
        w, h = xmax - xmin, ymax - ymin
    else:
        raise ValueError("bbox_format must be either 'xywh' or 'xyxy'")
    cropped_mask = None
    if mask is not None:
        cropped_mask = mask[ymin:ymax, xmin:xmax]
    
    # If a polygon mask is provided, adjust its coordinates relative to the crop.
    cropped_polygon = None
    if polygon_mask is not None:
        # Ensure the input is a numpy array of shape (N, 2)
        polygon_mask = np.asarray(polygon_mask)
        if polygon_mask.ndim != 2 or polygon_mask.shape[1] != 2:
            raise ValueError("polygon_mask must be a 2D array with shape (N_points, 2)")
        
        # Shift the polygon by the top-left corner of the bounding box.
        cropped_polygon = polygon_mask - np.array([xmin, ymin])
        
        cropped_polygon[:, 0] = np.clip(cropped_polygon[:, 0], 0, w)
        cropped_polygon[:, 1] = np.clip(cropped_polygon[:, 1], 0, h)
    
    return cropped_mask, cropped_polygon

def crop_dataset_by_label(dataset, images,target_label, crop_warning_level=logging.DEBUG):
    logger.setLevel(crop_warning_level)
    
    # determine the label labels for each of the files
    crop_labels = {}
    cropped_images = {}
    for file in dataset.files:
        
        if file.path not in crop_labels:
            crop_labels[file.path] = {
                'label': []
            }
        
        filtered_annotations = [
            annot for annot in file.annotations if annot.label_id == target_label
        ]
        
        if len(filtered_annotations) == 0:
            logger.warning(f'no label found in {file.path}')
            file.annotations = []
            continue
        
        if len(filtered_annotations) > 1:
            raise ValueError(f'more than one label found in {file.path}')
        
        crop_labels[file.path]['label'] = filtered_annotations[0].value.to_numpy()
    
    # delete all the empty files
    dataset.delete_empty_files()
    # delete the target label annotations
    dataset.delete_label(target_label)
    
    for file in dataset.files:
        if len(crop_labels[file.path]['label']) == 0:
            logger.warning(f'no label found in {file.path}')
            continue
        
        crop_box = crop_labels[file.path]['label'][:-1]
        cangle = crop_labels[file.path]['label'][-1]
        crop_box = np.array(crop_box).astype(np.int32)
        if cangle > 0:
            raise Exception(f'Obb is not supported')
        
        cx1, cy1, cx2, cy2 = crop_box   
        
        updated_annotations = []
        for annot in file.annotations:
            if annot.label_id == target_label:
                continue
            
            # Bounding Box Annotation
            if isinstance(annot, BoxAnnotation):
                x1, y1, x2, y2, angle = annot.value.to_numpy()
                #check if the box is inside the crop box
                if x1 < cx1 or y1 < cy1 or x2 > cx2 or y2 > cy2:
                    logger.warning(f'box {annot.id} is out of the crop box, skip')
                    continue
                # crop the box
                cropped_bbox = crop_bbox([cx1, cy1,cx2,cy2], [x1, y1, x2, y2])
                annotation = Box(
                    x_min=cropped_bbox[0],
                    y_min=cropped_bbox[1],
                    x_max=cropped_bbox[2],
                    y_max=cropped_bbox[3],
                    angle=angle,
                )
                updated_annotations.append(
                    BoxAnnotation(
                        id=annot.id,
                        value=annotation,
                        label_id=annot.label_id,
                        confidence=annot.confidence,
                        link=annot.link,
                        iou=annot.iou,
                    )
                )
            
            # Mask Annotation
            elif isinstance(annot, MaskAnnotation):
                mask = annot.value.to_numpy(h=file.height, w=file.width)
                cropped_mask, _ = crop_mask([cx1, cy1,cx2,cy2], mask=mask, bbox_format="xyxy")
                if cropped_mask is None:
                    logger.warning(f'mask {annot.id} is out of the crop box, skip')
                    continue
                annotation = Mask(
                    mask=cropped_mask,
                )
                updated_annotations.append(
                    MaskAnnotation(
                        id=annot.id,
                        value=annotation,
                        label_id=annot.label_id,
                        confidence=annot.confidence,
                        link=annot.link,
                        iou=annot.iou,
                    )
                )
            
            # Polygon Annotation
            elif isinstance(annot, PolygonAnnotation):
                polygon_points = annot.value.to_numpy()
                _, cropped_polygon = crop_mask([cx1, cy1,cx2,cy2], polygon_mask=polygon_points, bbox_format="xyxy")
                if cropped_polygon is None:
                    logger.warning(f'polygon {annot.id} is out of the crop box, skip')
                    continue
                annotation = Polygon(
                    points=cropped_polygon,
                )
                updated_annotations.append(
                    PolygonAnnotation(
                        id=annot.id,
                        value=annotation,
                        label_id=annot.label_id,
                        confidence=annot.confidence,
                        link=annot.link,
                        iou=annot.iou,
                    )
                )
            else:
                raise ValueError(f'unsupported annotation type {annot.type}')
        
        # save the cropped image
        image = images[file.path]
        if image is None:
            raise ValueError(f'failed to read {file.path}')
        
        cropped_image = image[cy1:cy2, cx1:cx2]
        cropped_images[file.path] = cropped_image
        
        # update the image size
        file.height = cropped_image.shape[0]
        file.width = cropped_image.shape[1]
        file.annotations = updated_annotations
        

        
    return cropped_images, dataset
        