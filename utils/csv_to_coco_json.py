import os
import glob
import cv2
import csv
import json
import collections
import numpy as np
from shapely.geometry import Polygon
from PIL import Image, ImageDraw

class Dataset(object):
    """
    create a coco format dataset from csv file
    """
    def __init__(self, path_pngs:str, path_csv:str, dt_category:dict, super_category:dict, json_out_path, plot=True) -> None:
        super().__init__()
        self.info = {
            "description": "Custom Dataset",
        }
        self.categories = []
        self.licenses = []
        self.images = []
        self.annotations = []
        self.imgfile2id = {}
        self.fname_to_fullpath = {}
        self.im_id = 1
        self.anno_id = 1

        #func
        self.add_categories(dt_category, super_category)
        self.add_imgs(path_pngs)
        self.add_annotations(path_csv, dt_category, plot)
        self.write_to_json(json_out_path)


    def add_categories(self, dt_category, super_category={}):
        """
        add categories for later writting to json
        arguments:
            dt_category(dict): the category dictionary <class name, id>
            super_category(dict): the super category dictionary
        """
        for cat in dt_category:
            dt = {}
            dt['supercategory'] = super_category[cat] if cat in super_category else ''
            dt['name'] = cat
            dt['id'] = dt_category[cat]
            self.categories.append(dt)
        

    def add_imgs(self, path_imgs):
        """
        add images from the path_img
        arguments:
            path_imgs(str): the path to image folder
        """
        files = glob.glob(os.path.join(path_imgs,'*.png'))
        for f in files:
            dt = {}
            im = cv2.imread(f)
            h,w = im.shape[:2]
            fname = os.path.basename(f)
            self.fname_to_fullpath[fname] = f
            dt['file_name'] = fname
            dt['height'] = h
            dt['width'] = w
            dt['id'] = self.im_id
            self.imgfile2id[fname] = self.im_id
            self.im_id += 1
            self.images.append(dt)


    def add_annotations(self, path_csv, dt_category, plot, iscrowd=False):
        """
        add annotations from csv, assume that each instance is not crowd
        arugments:
            path_csv(str): the path to csv file
            dt_category(dict): the mapping <category, id>
            plot(bool): wether to plot or not
            iscrowd(bool): wether the annotated instance is crowd or not
        """
        masks = {}
        rects = {}
        #read annotations from csv file
        with open(path_csv, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=';')
            for row in reader:
                fname = row[0]
                if row[2]=='polygon':
                    if fname not in masks:
                        masks[fname] = collections.defaultdict(list)
                    if row[3]=='x values':
                        if 'x' not in masks:
                            masks[fname]['x'].append(row[4:])
                    if row[3]=='y values':
                            masks[fname]['y'].append(row[4:])
                            masks[fname]['category'].append(row[1])
                            masks[fname]['iscrowd'].append(iscrowd)
                            masks[fname]['image_id'].append(self.imgfile2id[fname])
                elif row[2]=='rect':
                    if fname not in rects:
                        rects[fname] = collections.defaultdict(list)
                    if row[3]=='upper left':
                        bbox = row[4:][:]
                    if row[3]=='lower right':
                        bbox += row[4:]
                        rects[fname]['bbox'].append(bbox)
                        rects[fname]['category'].append(row[1])
                        rects[fname]['iscrowd'].append(iscrowd)
                        rects[fname]['image_id'].append(self.imgfile2id[fname])
        #generate coco annotations
        for fname in masks:
            mask = masks[fname]
            for x,y,cat_str,im_id,iscrowd in zip(mask['x'],mask['y'],mask['category'],mask['image_id'],mask['iscrowd']):
                #skip if category not in dictionary
                if cat_str not in dt_category:
                    continue
                
                dt = {}
                vertex = [float(v) for pt in zip(x,y) for v in pt] #(x1,y1,x2,y2)
                poly = Polygon([(float(xi),float(yi)) for xi,yi in zip(x,y)])

                if plot:
                    self.visualize(poly,fname)

                x_min,y_min,x_max,y_max = poly.bounds
                dt['segmentation'] = [vertex]
                dt['area'] = poly.area
                dt['iscrowd'] = iscrowd
                dt['image_id'] = im_id
                dt['bbox'] = [x_min,y_min,x_max-x_min+1,y_max-y_min+1]
                dt['category_id'] = dt_category[cat_str]
                dt['id'] = self.anno_id
                self.anno_id += 1
                self.annotations.append(dt)

        for fname in rects:
            rect = rects[fname]
            for bbox,cat_str,im_id,iscrowd in zip(rect['bbox'],rect['category'],rect['image_id'],rect['iscrowd']):
                #skip if category not in dictionary
                if cat_str not in dt_category:
                    continue

                dt = {}
                x1,y1,x2,y2 = [float(v) for v in bbox]
                w,h = x2-x1+1, y2-y1+1
                poly = Polygon([(x1,y1), (x2,y1), (x2,y2), (x1,y2)])

                if plot:
                    self.visualize(poly,fname)

                dt['segmentation'] = []
                dt['area'] = w*h
                dt['iscrowd'] = iscrowd
                dt['image_id'] = im_id
                dt['bbox'] = [x1,y1,w,h]
                dt['category_id'] = dt_category[cat_str]
                dt['id'] = self.anno_id
                self.anno_id += 1
                self.annotations.append(dt)


    def write_to_json(self, json_out_path):
        """
        write the whole dataset to coco json format
        arguments:
            json_out_path(str): the path to json output file
        """
        data = {
            'info': self.info, 'licenses': self.licenses, 
            'images': self.images, 'annotations': self.annotations,
            'categories': self.categories
            }
        with open(json_out_path, 'w') as f:
            json.dump(data, f)
        

    def visualize(self, polygon, fname):
        """
        visualize the polygon
        arguments:
            polygon(object): the polygon mask object
            fname(str): the file name
        """
        im = cv2.imread(self.fname_to_fullpath[fname])
        h,w = im.shape[:2]
        #create 2d polygon mask
        img = Image.new('L', (w, h), 0)
        ImageDraw.Draw(img).polygon(polygon.exterior.coords, outline=1, fill=1)
        mask = np.array(img)
        #if len(im.shape)>len(mask.shape):
        #    mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        mask = mask.astype(np.bool)
        #plot
        im = im.astype(np.float)
        im[~mask] *= 0.25
        cv2.imshow('plot',im.astype(np.uint8))
        cv2.waitKey(100)
                

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--path_img', required=True, type=str, help='the path to the images')
    ap.add_argument('--path_csv', required=True, type=str, help='the path to the csv file')
    ap.add_argument('--classes', required=True, type=str, help='the class categories in the dataset using comma to separate each category')
    ap.add_argument('--output_json', required=True, type=str, help='the path to the output json file')
    ap.add_argument('--plot', default=True, type=lambda x: x in ['True', 'true', '1'], help='plot the annotations')
    args = vars(ap.parse_args())

    #create a dictionary in this format: {'up':1, 'down':2}
    id = 1
    category = {}
    for cat in args['classes'].split(','):
        category[cat] = id
        id+=1
    Dataset(args['path_img'], args['path_csv'], dt_category=category, super_category={}, json_out_path=args['output_json'], plot=args['plot'])
