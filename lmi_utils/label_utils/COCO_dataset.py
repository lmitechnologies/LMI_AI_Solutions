from datetime import datetime
import cv2
import numpy as np
import json
import os
import glob
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def xyxy_to_xywh(x1,y1,x2,y2):
    return x1, y1, x2-x1, y2-y1
    
def xywh_to_xyxy(x,y,w,h):
    return x, y, x+w, y+h

def rotate(x,y,w,h,angle=0.0,rot_center='up_left',unit='degree'):
    """rotate the bbox from [x,y,w,h] using the angle to a array of [4,2].
    
    Args:
        angle(float): the rotation angle in current unit
        rot_center(str): the rotation center, either 'up_left' or 'center'
        unit(str): the current unit, either 'degree' or 'radian'. defalt unit is 'degree'

    Returns:
        np.ndarray: 4x2
    """
    if unit=='degree':
        ANGLE = np.deg2rad(angle)
    elif unit=='radian':
        ANGLE = angle
    else:
        raise Exception('Does not recognize the unit other than "degree" and "radian"')
    points = [[x,y],[x+w,y],[x+w,y+h],[x,y+h]]
    if rot_center=='up_left':
        xc,yc = x,y
    elif rot_center=='center':
        xc,yc = np.mean(points, axis=0)
    else:
        raise Exception('Does not recognize the rotation center other than "up_left" and "center"')
    return np.array(
        [
            [
                xc + np.cos(ANGLE) * (px - xc) - np.sin(ANGLE) * (py - yc),
                yc + np.sin(ANGLE) * (px - xc) + np.cos(ANGLE) * (py - yc)
            ]
            for px, py in points
        ]
    ).astype(int)



class Annotation:
    def __init__(self, path:str, category:str, bbox:list, segmentation=[], rotation=None) -> None:
        """init Annotation class for coco format

        Args:
            fname (str): the path to the file
            bbox (list): [x,y,w,h]
            segmentation (list, optional): _description_. Defaults to [].
            rotation (_type_, optional): _description_. Defaults to None.
        """
        self.path = path
        self.bbox = bbox
        self.category = category
        # assign later
        self.image_id = None
        self.category_id = None
        self.id = None
        self.iscrowd = None
        self.area = None
        
        # optional
        self.segmentation = segmentation
        self.rotation = rotation
        
        
    def __str__(self) -> str:
        return f'path: {self.path}\nbbox: {self.bbox}\nrotation: {self.rotation}\n'+\
            f'category: {self.category}\nimage_id: {self.image_id}\ncategory_id: {self.category_id}'
    
    
    def to_dt(self):
        """convert the annotation to a dictionary
        add rotation angle as the 5th element in bbox

        Returns:
            dict: the dictionary for output to a json file 
        """
        if self.image_id is None or self.category_id is None:
            raise Exception('image id or cateogry id is None')
        dt = {}
        dt['id'] = self.id
        dt['image_id'] = self.image_id
        dt['category_id'] = self.category_id
        dt['iscrowd'] = self.iscrowd
        if isinstance(self.bbox, np.ndarray):
            self.bbox = self.bbox.tolist()
        dt['bbox'] = self.bbox
        dt['area'] = self.area
        if len(self.segmentation):
            if isinstance(self.segmentation, np.ndarray):
                self.segmentation = self.segmentation.tolist()
            dt['segmentation'] = self.segmentation
        if self.rotation is not None:
            dt['bbox'] += [self.rotation]
        return dt



class COCO_Dataset:
    def __init__(self, dt_category:dict, path_imgs:str) -> None:
        """create coco dataset which supports rotated bbox

        Args:
            dt_category (dict): the category dictionary <class name, id>
            path_imgs (str): the path to images in the dataset
        """
        self.info = {
            "description": "Custom Dataset",
            "date_created": datetime.today().strftime('%Y/%m/%d')
        }
        self.dt_category = dt_category
        self.licenses = []
        self.images = []
        self.annotations = []
        self.categories = []
        self.imgfile2id = {}
        self.im_id = 1
        self.annot_id = 1
        
        self.add_categories()
        self.add_imgs(path_imgs)
        
        
    def add_categories(self, super_category={}):
        """
        add categories for later writting to json
        arguments:
            dt_category(dict): the category dictionary <class name, id>
            super_category(dict): the super category dictionary
        """
        for cat in self.dt_category:
            dt = {}
            dt['supercategory'] = super_category[cat] if cat in super_category else ''
            dt['name'] = cat
            dt['id'] = self.dt_category[cat]
            self.categories.append(dt)
    
    
    def add_imgs(self, path_imgs, fmts=['png','jpeg','jpg']):
        """
        add images from the path_img
        arguments:
            path_imgs(str): the path to image folder
            fmts(list): the list of image formats
        """
        files = []
        for fmt in fmts:
            files += glob.glob(os.path.join(path_imgs,'*.'+fmt))
        logger.info(f'found {len(files)} images in {path_imgs}')
        for f in files:
            dt = {}
            im = cv2.imread(f)
            h,w = im.shape[:2]
            fname = os.path.basename(f)
            dt['file_name'] = fname
            dt['height'] = h
            dt['width'] = w
            dt['id'] = self.im_id
            self.imgfile2id[fname] = self.im_id
            self.im_id += 1
            self.images.append(dt)
            
    
    def add_annotations(self, annots:list, plot=False):
        """add annotations from a list of Annotation class objects

        Args:
            annots (list): a list of Annotation class objects
            plot (bool, optional): plot the annotations. Defaults to False.
        """
        logger.info(f'adding {len(annots)} annotations to the coco dataset')
        for annot in annots:
            self.__add_annotation(annot,plot)
            
            
    def __add_annotation(self, annot:Annotation, plot=False):
        """
        add annotation
        arugments:
            annot(Annotation): the Annotation class object
            plot(bool): plot the annotation
        """
        self.assign_ids_to_annot(annot)
        self.annotations.append(annot.to_dt())
        
        if plot:
            self.visualize(annot)
    
    
    def assign_ids_to_annot(self, annot:Annotation):
        """assign image and category ids to annot

        Args:
            annot (Annotation): the Annotation class object
        """
        fname = os.path.basename(annot.path)
        if fname not in self.imgfile2id:
            raise Exception('must call add_imgs() before add_annotations() or add_annotation()')
        img_id = self.imgfile2id[fname]
        annot.image_id = img_id
        annot.category_id = self.dt_category[annot.category]
        annot.iscrowd = 0
        annot.id = self.annot_id
        self.annot_id += 1
    
            
    def visualize(self, annot:Annotation, ratio=0.25):
        """
        visualize the annotation. Currently, ONLY plot bbox
        arguments:
            pts(list): a list of [x,y]
            fname(str): the file name
        """
        img = cv2.imread(annot.path)
        pts = rotate(*annot.bbox)
        cv2.drawContours(img, [pts], 0, (0, 255, 0), 3, cv2.LINE_AA)
        if len(annot.segmentation):
            segs = np.array(annot.segmentation).astype(int)
            cv2.drawContours(img, [segs], 0, (0, 255, 0), 3, cv2.LINE_AA)
        img2 = cv2.resize(img, (0,0), fx=ratio, fy=ratio)
        cv2.imshow('plot',img2)
        cv2.waitKey(100)
        
        
    def write_to_json(self, out_path):
        """
        write the whole dataset to coco json format
        arguments:
            out_path(str): the path to json output file
        """
        data = {
            'info': self.info, 'licenses': self.licenses, 
            'images': self.images, 'annotations': self.annotations,
            'categories': self.categories
            }
        with open(out_path, 'w') as f:
            json.dump(data, f)
