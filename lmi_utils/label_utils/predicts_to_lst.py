from PIL import Image
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
    cmd = ["gsutil", "-m", "cp", "-r", "gs://" + bucket, "./data"]
    print(f'cmd: {cmd}')
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
                                #'score': 0.5,   #TODO - load score form pipeline
                                'rotation': 0,
                                'rectanglelabels': [boxes['object']]
                            },
                            'from_name': 'rectangle',
                            'to_name': 'image',
                            'type': 'rectanglelabels'
                        }
                        label_obj['predictions'][0]['result'].append(box)
                
                if 'polygons' in label_json:
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
                                ],
                                #'score': 0.5,   #TODO - load score form pipeline
                            },
                            'from_name': 'polygon',
                            'to_name': 'image',
                            'type': 'polygonlabels'
                        }
                        for i in range(0, len(polygons['x'])):
                            x = polygons['x'][i] / width * 100
                            y = polygons['y'][i] / height * 100
                            polygon['value']['points'].append([x,y])
                        
                        label_obj['predictions'][0]['result'].append(polygon)

            labels.append(label_obj)


    if not os.path.isdir(destination):
        os.mkdir(destination)

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
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', '-i', required=True, help='location of the GCP data storage path. Ex: bucket_name/folder_name/...')
    ap.add_argument('--dest', '-o', required=True, help='location results should be put')
    args=ap.parse_args()

    # download_data_from_bucket(args.src)
    data_path = Path("./data") / Path(args.src).stem
    files = get_files(data_path)
    convert_to_ls(files, args.dest, args.src)
    # delete_data()
