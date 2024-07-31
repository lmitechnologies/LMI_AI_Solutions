from PIL import Image
from pathlib import Path
import json
import subprocess
import os
import logging
from label_utils.csv_to_lst import write_xml, RECT_NAME, POLYGON_NAME

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


"""
Label data should be formatted like this:

    label_json = {
        'image_width': width,
        'image_height': height,
        'boxes': [
            {
                'object': DEFECT_NAME,
                'x': X,             // top left
                'y': Y,             // top left
                'width': width,     // box width
                'height': height,   // box height
                'score': score,
                'rotation': angle,  // box rotation angle in degrees
            }
        ],
        'polygons': [
            {
                'object': DEFECT_NAME,
                'x': [X1, X2, X3, X4],
                'y': [Y1, Y2, Y3, Y4],
                'score': score,
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

The label data needs to be called "labels".
    
"""


def download_data_from_bucket(bucket):
    if not os.path.isdir("./data"):
        os.mkdir("./data")
    cmd = ["gsutil", "-m", "cp", "-r", "gs://" + bucket, "./data"]
    logger.info(f'cmd: {cmd}')
    subprocess.run(cmd)


def get_files(source):
    files = {}
    result = [os.path.join(dp, f) for dp, dn, filenames in os.walk(source) for f in filenames]
    for file in result:
        lists = file.split(os.sep)
        if 'pipeline' in lists and 'labels' in lists:
            type = 'labels'
        elif 'sensor' in lists and 'image' in lists:
            type = 'image'
        else:
            continue
        path = Path(file)
        id = path.stem.split('.')[0]
        if files.get(id) is not None:
            files[id][type] = path
        else:
            files[id] = {
                type: path
            }
        if type == 'image':
            files[id]['size'] = path.stat().st_size
    return files        


def convert_to_ls(files, destination, bucket):
    """
    Output: A label.json file to be imported in Label Studio
    """
    labels = []
    box_class = set()
    polygon_class = set()
    cnt_box = 0
    cnt_polygon = 0
    for key in files:
        if 'image' not in files[key].keys() or 'labels' not in files[key].keys():
            continue
        
        img = Image.open(files[key]['image'])
        width, height = img.size
        with open(files[key]['labels'], 'r') as file:
            label_json = json.load(file)
            img_path = "gs://" + bucket + str(files[key]['image']).split(Path(bucket).stem)[1].replace("\\", "/")
            label_obj = {
                'predictions': [
                    {
                        'model_version': 'pipeline_prediction',
                        'result': []
                    }
                ],
                'data': {
                    'image': img_path
                }
            }

            if 'boxes' in label_json:
                for box in label_json['boxes']:
                    box_class.add(box['object'])
                    box_lst = {
                        "original_width": width,
                        "original_height": height,
                        "image_rotation": 0,
                        'value': {
                            'x': box['x'] / width * 100,
                            'y': box['y'] / height * 100,
                            'width': box['width'] / width * 100,
                            'height': box['height'] / height * 100,
                            'score': box.get('score', None),
                            'rotation': box.get('rotation', 0),
                            'rectanglelabels': [box['object']]
                        },
                        'from_name': RECT_NAME,
                        'to_name': 'image',
                        'type': 'rectanglelabels'
                    }
                    label_obj['predictions'][0]['result'].append(box_lst)
                    cnt_box += 1
            
            if 'polygons' in label_json:
                for polygon in label_json['polygons']:
                    polygon_class.add(polygon['object'])
                    polygon_lst = {
                        "original_width": width,
                        "original_height": height,
                        "image_rotation": 0,
                        'value': {
                            'points': [],
                            'polygonlabels': [
                                polygon['object']
                            ],
                            'score': polygon.get('score', None),
                        },
                        'from_name': POLYGON_NAME,
                        'to_name': 'image',
                        'type': 'polygonlabels'
                    }
                    for i in range(len(polygon['x'])):
                        x = polygon['x'][i] / width * 100
                        y = polygon['y'][i] / height * 100
                        polygon_lst['value']['points'].append([x,y])
                    
                    label_obj['predictions'][0]['result'].append(polygon_lst)
                    cnt_polygon += 1

        labels.append(label_obj)

    if cnt_box == 0 and cnt_polygon == 0:
        logger.warning("No labels found, skip!")
        return

    if not os.path.isdir(destination):
        os.mkdir(destination)
        
    label_path = Path(f"{destination}/label.json")
    with open(label_path, 'w') as file:
        json.dump(labels, file, indent=4)
        
    write_xml(destination, box_class, polygon_class)



if __name__=="__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', '-i', required=True, help='location of the GCP data storage path. Ex: bucket_name/folder_name/...')
    ap.add_argument('--dest', '-o', required=True, help='location results should be put')
    args=ap.parse_args()

    download_data_from_bucket(args.src)
    data_path = Path("./data") / Path(args.src).stem
    files = get_files(data_path)
    convert_to_ls(files, args.dest, args.src)
