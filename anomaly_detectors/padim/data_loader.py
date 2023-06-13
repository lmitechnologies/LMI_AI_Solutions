"""
    MODULE: data_loader.py

    USAGE:
        load data from directory and subdirectories
            
"""
# 1. Built-in modules
import os
import glob

# 2. Third-party modules
import tensorflow as tf

class DataLoader(object):
    """
    DESCRIPTION:
        loads images from directory and subdirectories into tf.data.Dataset iterable.
        Note: ONLY load 8 bits PNG images.
    """
    def __init__(self, path_base, img_shape, batch_size, normalize=False, shuffle=True, random_flip_h=False, random_flip_v=False, img_types=['png']):
        """
        DESCRIPTION:
            1. set the image shape
            2. read and save the image filenames into a list
            3. generate the tf.data.Dataset iterable: 
            It will resize the image size according to the im_shape, and normalize between [0,1] ONLY if normalize is True
        ARGUMENTS:
            path_base -> a string for the base path of data files and or directories containing data files
            img_shape -> a tuple of the image shape
            batch_size -> a int for batch size
            normalize -> a bool for normalizing the image or not
            shuffle -> a bool for shuffling the dataset or not
        MODIFIES:
            self.dataset: a td.data.Dataset iterable
            self.img_shape: a tuple of image shape, (width, height)
            self.file_list: a list of full path of image files to be loaded
            self.names: a list of image files names to be loaded
        """

        self.img_shape = img_shape
        self.normalize = normalize

        #get image file list from path_base and its subfolders
        self.file_list, self.file_names = self._get_file_list(path_base, img_types=img_types)
        self.n_samples = len(self.file_list)

        #generate dataset from the file list
        dataset = tf.data.Dataset.from_tensor_slices((self.file_list, self.file_names))

        if shuffle:
            dataset = dataset.shuffle(self.n_samples, reshuffle_each_iteration=True)

        lambda_parse=lambda path_file, file_name: self._parse_function(path_file, file_name,random_flip_h, random_flip_v)

        #apply the parse function to each element in the dataset
        dataset = dataset.map(lambda_parse, num_parallel_calls=tf.data.AUTOTUNE)

        #set batch size
        dataset = dataset.batch(batch_size)

        #prefetch for sppeedup
        self.dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
    @staticmethod
    def _get_file_list(path_base, img_types):
        """
        DESCRIPTION:
            get image file list from path_base and its subfolders
        ARGUMENTS:
            path_base -> a string for path base
        RETURNS:
            file_list -> a list of full paths to image files
            file_names -> a list of image file names
        """
        file_list = []
        file_names = []
        dirs = os.listdir(path_base)
        subdirs = [dirx for dirx in dirs if os.path.isdir(os.path.join(path_base,dirx)) ]
        if not subdirs:
            subdirs=['']
        # concatenate all the file lists from subfolders
        for subdir in subdirs:
            path = os.path.join(path_base,subdir)
            cur_list = []
            for img_type in img_types:
                cur_list.extend(glob.glob(os.path.join(path, f'*.{img_type}')))
            fnames = [os.path.basename(l) for l in cur_list]
            file_list += cur_list
            file_names += fnames
        return file_list, file_names

    def _parse_function(self, path_file, file_name, random_flip_h, random_flip_v):
        """
        DESCRIPTION:
            parse each element in the dataset
            1. read the content of image
            2. decode it
            3. if self.normalize is True, convert it to float between [0,1]
            4. resize to shape
        ARGUMENTS:
            path_file -> a string of the full path to the image file
            file_name -> a string of the image file name 
        RETURNS:
            image -> a 3D tf.Tensor for the image
            file_name -> a string of the image file name 
        """
        print(f'[INFO] Loading data from: {path_file} for {file_name}')
        

        raw = tf.io.read_file(path_file)
        #loads the image as a uint16 tensor. No losses when uint8 is converted to uint16 tensor
        # works for both uint16 and uint8 images
        image = tf.io.decode_image(raw, expand_animations=False, dtype=tf.dtypes.uint16)

        if tf.shape(image)[-1] == 1:
            image = tf.image.grayscale_to_rgb(image)
        
        if self.normalize:
            #convert to float values in [0,1]
            image = tf.image.convert_image_dtype(image, tf.float32)
        else:
            #convert to float
            image = tf.cast(image, dtype=tf.float32)

        #resize image
        image = tf.image.resize(image, size=self.img_shape, method='bicubic')

        if random_flip_h:
            image=tf.image.random_flip_left_right(image)
        if random_flip_v:
            image=tf.image.random_flip_up_down(image)
        
        return image, file_name


if __name__ == '__main__':
    import argparse as ap
    import cv2
    import numpy as np

    parser = ap.ArgumentParser()
    parser.add_argument('--path',default='./data/padim/expected')
    args=parser.parse_args()

    dataloader = DataLoader(args.path, img_shape=(224,224), batch_size=32, normalize=False, shuffle=True)

    for images,fnames in dataloader.dataset:
        print(images[0,100:200,100:200,:])
        for img,fname in zip(images.numpy().astype(np.uint8), fnames.numpy()):
            print(f"[INFO] Showing file: {fname}")
            cv2.imshow('Test Img',cv2.cvtColor(img,cv2.COLOR_RGB2BGR))
            cv2.waitKey(500)
    
    cv2.destroyAllWindows()