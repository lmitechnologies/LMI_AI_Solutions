import itertools
import os
import io
import cv2
from PIL import Image
import tensorflow as tf
import numpy as np
import pandas as pd
import time
from tqdm import tqdm
import argparse
from object_detection.inference import detection_inference
from object_detection.utils.dataset_util import bytes_list_feature
from object_detection.utils.dataset_util import float_list_feature
from object_detection.utils.dataset_util import int64_list_feature
from object_detection.utils.dataset_util import int64_feature

from object_detection.utils.dataset_util import bytes_feature
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.core import data_parser
from object_detection.core import standard_fields as fields

from object_detection.metrics.tf_example_parser import BoundingBoxParser, StringParser, Int64Parser, FloatParser

# TODO: 1/18/2021 need testing for classifier and segmentation versions
class CustomParser(data_parser.DataToNumpyParser):
    """
    Tensorflow Example proto parser.

    Application type must be one of 'detection', 'segmentation', 'classification'
    """

    def __init__(self, application_type='detection', allow_missing=False):
        self.items_to_handlers = {
            fields.InputDataFields.image: StringParser(
                        fields.TfExampleFields.image_encoded),
            fields.InputDataFields.filename: StringParser(fields.TfExampleFields.filename)
        }
        if application_type == 'detection':
            self.items_to_handlers[fields.InputDataFields.groundtruth_boxes] = BoundingBoxParser(
                fields.TfExampleFields.object_bbox_xmin,
                fields.TfExampleFields.object_bbox_ymin,
                fields.TfExampleFields.object_bbox_xmax,
                fields.TfExampleFields.object_bbox_ymax
            )
            self.items_to_handlers[fields.InputDataFields.groundtruth_classes] = Int64Parser(
                fields.TfExampleFields.object_class_label
            )

        elif application_type == 'segmentation':
            self.items_to_handlers[fields.InputDataFields.groundtruth_instance_masks] = StringParser(
                fields.TfExampleFields.instance_masks
            )
            # TODO: tfannotation.py does not yet have an instance_classes field. Relying on implicit classes and requires allow_missing True
            self.items_to_handlers[fields.InputDataFields.groundtruth_instance_classes] = StringParser(
                fields.TfExampleFields.instance_classes
            )

        elif application_type == 'classification':
            self.items_to_handlers[fields.InputDataFields.groundtruth_image_classes] = Int64Parser(
                fields.TfExampleFields.image_class_label
            )
        else:
            raise ValueError('Application type must be one of [detection, segmentation, classification]')
        
    def parse(self, tf_example, allow_missing=False):
        """
        Parses tensorflow example and returns a tensor dictionary.
        Args:
            tf_example: a tf.Example object.
            allow_missing: whether or not to allow missing fields, will throw warning instead of error
        Returns:
            A dictionary of the following numpy arrays:
            image               - string containing input image.
            filename            - string containing input filename (optional, None if not specified)
            groundtruth_boxes   - a numpy array containing groundtruth boxes.
            groundtruth_classes - a numpy array containing groundtruth classes.
        """
        results_dict = {}
        missing_keys = []
        parsed = True
        for key, parser in self.items_to_handlers.items():
            results_dict[key] = parser.parse(tf_example)
            if results_dict[key] is None:
                parsed = False
                missing_keys.append(key)

        if not parsed:
            print(f'[WARNING] The following fields were not found in the input tfrecord file:\n{missing_keys}')
            if not allow_missing:
                results_dict = None
        return results_dict

def parse_function(record, record_parser):
    '''
    Parses out image, groundtruth boxes
    '''
    example = tf.train.Example()
    example.ParseFromString(record.numpy())
    decoded_dict = record_parser.parse(example)
    return decoded_dict
    

if __name__ == '__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('-i', '--input_record',type=str,required=True)
    ap.add_argument('-o', '--output_data_path',type=str,required=True)
    ap.add_argument('--application_type',type=str,default='detection')
    ap.add_argument('--allow_missing', dest='allow_missing', action='store_true')
    ap.set_defaults(allow_missing=False)

    args=vars(ap.parse_args())

    inp = args['input_record']
    out = args['output_data_path']
    at  = args['application_type']
    am  = args['allow_missing']

    record_parser = CustomParser(at, am)
    dataset = tf.data.TFRecordDataset(inp)

    for i, record in enumerate(tqdm(dataset)):
        decoded_dict = parse_function(record, record_parser)
        # TODO: nice-to-have would be to also reverse write classes and such back to csv/json
        filename = decoded_dict[fields.InputDataFields.filename].decode('utf-8')
        if not filename:
            filename = f'image_{i}.png'
        image = Image.open(io.BytesIO(decoded_dict[fields.InputDataFields.image]))
        image = np.array(image)
        outpath = os.path.join(out, filename)
        image=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
        cv2.imwrite(outpath, image)
    
    print('Done!')


    


    
