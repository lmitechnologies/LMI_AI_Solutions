import tensorflow as tf
import glob
import os
import numpy as np

import sys
sys.path.append('/app/LMI_AI_Solutions/lmi_utils')
sys.path.append('/app/LMI_AI_Solutions/object_detectors')
sys.path.append('/app/LMI_AI_Solutions/object_detectors/tf_objdet/models/research')
sys.path.append('/app/LMI_AI_Solutions/acceleration/tensorrt/tf/tensorrt/tftrt/benchmarking-python')

from benchmark_args import BaseCommandLineAPI
from benchmark_runner import BaseBenchmarkRunner

class CommandLineAPI(BaseCommandLineAPI):

    def __init__(self):
        super(CommandLineAPI, self).__init__()

        self._parser.add_argument(
            '--input_size',
            type=int,
            default=640,
            help='Size of input images expected by the '
            'model'
        )

        self._add_bool_argument(
            name="benchmark_original",
            default=False,
            required=False,
            help="If set to True, will benchmark original saved model."
        )

        self._add_bool_argument(
            name="benchmark_trt",
            default=False,
            required=False,
            help="If set to True, will benchmark trt saved model."
        )

        

class BenchmarkRunner(BaseBenchmarkRunner):
    def get_dataset_batches(self):
        ''' Required Base Class Method.

            load images from directory
            convert to tf dataset
            apply image preprocessing (resize, padding, etc)
            return dataset
        '''
        def _preproc_function(file_path):
            raw = tf.io.read_file(file_path)
            # loads the image as a uint8 tensor
            image = tf.io.decode_image(raw, expand_animations=False)
            # assumes no resizing/padding
            image,_=self.preprocess_model_inputs(image)
            return image
            
        types = ('*.png', '*.jpg')
        file_list = []
        for file_type in types:
            file_list.extend(glob.glob(os.path.join(self._args.data_dir,file_type)))
        dataset=tf.data.Dataset.from_tensor_slices(file_list)
        dataset = dataset.map(_preproc_function, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(self._args.batch_size, drop_remainder=False)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

        return dataset, None

    def preprocess_model_inputs(self,image):
        """ Required Base Class Method.
        
        This function prepare the `data_batch` generated from the dataset.
        Returns:
            x: input of the model
            y: data to be used for model evaluation

        Note: script arguments can be accessed using `self._args.attr`
        """
        img_shape=image.shape
        if len(img_shape)==4:
            if image.shape[3]==1:
                image=tf.image.grayscale_to_rgb(image)
        elif len(img_shape)==3:
            if image.shape[2]==1:
                image=tf.image.grayscale_to_rgb(image)
        elif len(img_shape)==2:
            image=tf.expand_dims(image, -1)
            image=tf.image.grayscale_to_rgb(image)
        return image, np.array([])


    def postprocess_model_outputs(self,predictions,expected):
        """Required Base Class Method.

        Post process if needed the predictions and expected tensors. At the
        minimum, this function transforms all TF Tensors into a numpy arrays.
        Most models will not need to modify this function.

        Note: script arguments can be accessed using `self._args.attr`
        """

        predictions = {k: t.numpy() for k, t in predictions.items()}

        return predictions,expected

    def evaluate_model(self, predictions, expected, bypass_data_to_eval):
        """Required Base Class Method.

        Evaluate result predictions for entire dataset.

        This computes overall accuracy, mAP,  etc.  Returns the
        metric value and a metric_units string naming the metric.

        Note: script arguments can be accessed using `self._args.attr`
        """
        return 0,"None"

if __name__=='__main__':
    cmdline_api = CommandLineAPI()
    args = cmdline_api.parse_args()

    runner = BenchmarkRunner(args)

    runner.execute_benchmark()

