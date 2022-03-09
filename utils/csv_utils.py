#built-in packages
import csv
import collections
from logging import warning
import os

#LMI packages
import mask
import rect


def load_csv(fname:str, path_img:str, class_map:dict=None, zero_index:bool=True):
    """
    load csv file into a dictionary mapping <image_name, a list of mask objects>
    Arguments:
        fname(str): the input csv file name
        path_img(str): the path to the image folder where its images should be listed in the csv file
        class_map(dict): map <class, class ID>
	    zero_index(bool): it's used when class_map is None. whether the class ID is 0 or 1 indexed, default is True.
    Return:
        a dictionary maps <image_name, a list of Mask or Rect objects>
	    class_map(dict): <classname, ID> where IDs are 0-indexed if zero_index is true else 1-indexed.
    """
    masks = collections.defaultdict(list)
    rects = collections.defaultdict(list)
    if class_map is None:
        new_map = True
        class_map = {}
        id = 0 if zero_index else 1
    else:
        new_map = False
        id = max(class_map.values())+1

    with open(fname, newline='') as f:
        reader = csv.reader(f, delimiter=';')
        for row in reader:
            im_name = row[0]
            category = row[1]
            confidence = row[2]
            shape_type = row[3]
            coord_type = row[4]
            coordinates = row[5:]
            if category not in class_map:
                if not new_map:
                    warning(f'found new class in the csv: {category}')
                class_map[category] = id
                id += 1

            if shape_type=='polygon':
                if coord_type=='x values':
                    M = mask.Mask(im_name=im_name, fullpath=os.path.join(path_img,im_name), category=category, confidence=confidence)
                    M.X = list(map(int,coordinates))
                elif coord_type=='y values':
                    assert(im_name==M.im_name)
                    M.Y = list(map(int,coordinates))
                    masks[im_name].append(M)
                else:
                    raise Exception("invalid keywords: {}".format(coord_type))
            elif shape_type=='rect':
                if coord_type=='upper left':
                    R = rect.Rect(im_name=im_name, fullpath=os.path.join(path_img,im_name), category=category, confidence=confidence)
                    R.up_left = list(map(int,coordinates))
                elif coord_type=='lower right':
                    assert(im_name==R.im_name)
                    R.bottom_right = list(map(int,coordinates))
                    rects[im_name].append(R)
                else:
                    raise Exception("invalid keywords: {}".format(coord_type))
    return masks if masks else rects, class_map


def write_to_csv(shapes:dict, filename:str):
    """
    write a dictionary of list of shapes into a csv file
    Arguments:
        shape(dict): a dictionary maps the filename to a list of Mask or Rect objects, i.e., <filename, list of Mask or Rect>
        filename(str): the output csv filename
    """
    with open(filename, 'w') as f:
        writer = csv.writer(f, delimiter=';')
        for im_name in shapes:
            for shape in shapes[im_name]:
                if isinstance(shape, rect.Rect):
                    writer.writerow([shape.im_name, shape.category, f'{shape.confidence:.4f}', 'rect', 'upper left'] + shape.up_left)
                    writer.writerow([shape.im_name, shape.category, f'{shape.confidence:.4f}', 'rect', 'lower right'] + shape.bottom_right)
                elif isinstance(shape, mask.Mask):
                    writer.writerow([shape.im_name, shape.category, f'{shape.confidence:.4f}', 'polygon', 'x values'] + shape.X)
                    writer.writerow([shape.im_name, shape.category, f'{shape.confidence:.4f}', 'polygon', 'y values'] + shape.Y)
                else:
                    raise Exception("Found unsupported classes. Supported classes are mask and rect")
                    
