from PIL import Image
import tarfile
from pathlib import Path
import shutil
import json
import subprocess
import os

"""
Label data should be formatted like this:
    {
        'boxes': [
            {
                'object': DEFECT_NAME,
                'x': X,     // top left
                'y': Y,     // top left
                'x1': X1,   // bottom right
                'y1': Y1,   // bottom right
                'width': WIDTH,
                'height': HEIGHT
            }
        ]
        'polygons': [
            {
                'object': DEFECT_NAME,
                'x': [X1, X2, X3, X4],
                'y': [Y1, Y2, Y3, Y4]
            }
        ]
    }

The pipeline results should look like this:

    result['outputs'] = { 
            'annotated': inputs['image'],
            'labels': {
                'type': 'object',
                'format': 'json',
                'content': label_json
            }   
    }

The label data needs to be called "labels"
    
"""

def download_data_from_bucket(bucket):
    if not os.path.isdir("./data"):
        os.mkdir("./data")
    subprocess.call(["gsutil", "-m", "cp", "-r", "gs://" + bucket, "./data"])

def delete_data():
    shutil.rmtree("./data")

def get_files(source):
    files = {}
    result = [os.path.join(dp, f) for dp, dn, filenames in os.walk(source) for f in filenames]
    for file in result:
        if 'annotated' in file or 'log.txt' in file:
            continue 
        path = Path(file)
        id = path.stem.split('.')[0]
        type = 'labels' if 'pipeline' in file else 'image'
        if files.get(id) is not None:
            files[id][type] = path
        else:
            files[id] = {
                type: path
            }
        if type == 'image':
            files[id]['size'] = path.stat().st_size
    
    return files

def convert_to_vgg(files, destination):
    """
    Output: folder full of images and a label.json file
    User loads the images into VGG and imports the label.json 
    """
    labels = {}
    for key in files:
        if 'image' in files[key].keys():
            src_path = files[key]['image']
            dest_path = Path(destination) / files[key]['image'].name
            shutil.copyfile(src_path, dest_path)

            if 'labels' in files[key].keys():
                with open(files[key]['labels'], 'r') as file:
                    label_json = json.load(file)

                label_obj = {
                    "filenames": dest_path.name,
                    "size": files[key]['size'],
                    "regions": [],
                    "file_attributes": {}
                }


                for boxes in label_json['boxes']:
                    label_obj['regions'].append({
                        'shape_attributes': {
                            'name': 'rect',
                            'x': int(boxes['x']),
                            'y': int(boxes['y']),
                            'width': int(boxes['width']),
                            'height': int(boxes['height'])
                        },
                        'region_attributes': {
                            'object': boxes['object']
                        }
                    })

                for polygon in label_json['polygons']:
                    label_obj['regions'].append({
                        'shape_attributes': {
                            'name': 'polygon',
                            'all_points_x': polygon['x'],
                            'all_points_y': polygon['y']
                        },
                        'region_attributes': {
                            'object': polygon['object']
                        }
                    })

                label_key = str(files[key]['image'].name) + str(files[key]['size'])
                labels[label_key] = label_obj

    label_path = Path(f"{destination}/label.json")
    with open(label_path, 'w') as file:
        json.dump(labels, file, indent=4)            


def convert_to_ls(files, destination, bucket):
    """
    Output: A label.json file and labeling_interface.html file
    User copies the contents of labeling_interface.html into the Labeling Interface code editor
    then import label.json
    """
    labels = []
    box_types = set()
    polygon_types = set()
    for key in files:
        if 'image' in files[key].keys() and 'labels' in files[key].keys():
            img = Image.open(files[key]['image'])
            width, height = img.size
            with open(files[key]['labels'], 'r') as file:
                label_json = json.load(file)
                label_obj = {
                    'annotations': [
                        {
                            'result': []
                        }
                    ],
                    'data': {
                        'image': "gs://" + bucket + str(files[key]['image']).split(Path(bucket).stem)[1]
                    }
                }

                for boxes in label_json['boxes']:
                    box_types.add(boxes['object'])
                    box = {
                        "original_width": width,
                        "original_height": height,
                        "image_rotation": 0,
                        'value': {
                            'x': boxes['x'] / width * 100,
                            'y': boxes['y'] / height * 100,
                            'width': boxes['width'] / width * 100,
                            'height': boxes['height'] / height * 100,
                            'rotation': 0,
                            'rectanglelabels': [boxes['object']]
                        },
                        'from_name': 'rectangle',
                        'to_name': 'image',
                        'type': 'rectanglelabels'
                    }
                    label_obj['annotations'][0]['result'].append(box)
                
                for polygons in label_json['polygons']:
                    polygon_types.add(polygons['object'])
                    polygon = {
                        "original_width": width,
                        "original_height": height,
                        "image_rotation": 0,
                        'value': {
                            'points': [],
                            'polygonlabels': [
                                polygons['object']
                            ]
                        },
                        'from_name': 'polygon',
                        'to_name': 'image',
                        'type': 'polygonlabels'
                    }
                    for i in range(0, len(polygons['x'])):
                        x = polygons['x'][i] / width * 100
                        y = polygons['y'][i] / height * 100
                        polygon['value']['points'].append([x,y])
                    
                    label_obj['annotations'][0]['result'].append(polygon)

            labels.append(label_obj)

    label_path = Path(f"{destination}/label.json")
    with open(label_path, 'w') as file:
        json.dump(labels, file, indent=4)

    labeling_interface = "<View>\n"
    labeling_interface += "  <Header value=\"Select label and click the image to start\"/>\n"
    labeling_interface += "  <Image name=\"image\" value=\"$image\" zoom=\"true\"/>\n"
    labeling_interface += "  <RectangleLabels name=\"rectangle\" toName=\"image\">\n"
    for name in box_types:
        labeling_interface += f"    <Label value=\"{name}\" background=\"green\"/>\n"
    labeling_interface += "  </RectangleLabels>"
    labeling_interface += "  <PolygonLabels name=\"polygon\" toName=\"image\" strokeWidth=\"3\" pointSize=\"small\" opacity=\"0.9\">\n"
    for name in polygon_types:
        labeling_interface += f"    <Label value=\"{name}\" background=\"blue\"/>\n"
    labeling_interface += "  </PolygonLabels>\n"
    labeling_interface += "</View>"

    html_path = Path(f"{destination}/labeling_interface.html")
    with open(html_path, 'w') as file:
        file.write(labeling_interface)


if __name__=="__main__":
    # import argparse
    # ap=argparse.ArgumentParser()
    # ap.add_argument('--option', required=True, help='vgg or ls')
    # ap.add_argument('--src', required=True, help='location of the tar file')
    # ap.add_argument('--dest', required=True, help='location results should be put')
    
    # args=vars(ap.parse_args())
    # option=args['option']
    # src=args['src']
    # dest=args['dest']
    # bucket=args['bucket']

    # if option == 'vgg':
    #     convert_to_vgg(src, dest)
    # elif option == 'ls':
    #     convert_to_ls(src, dest, bucket)
    src = 'lmi-factorysmart-dev-3-data/Winchester-Board-232/archive/archive_2023-12-08T19-29-32-981Z'
    download_data_from_bucket(src)
    data_path = Path("./data") / Path(src).stem
    files = get_files(data_path)
    convert_to_ls(files, "./dest", src)
    delete_data()
