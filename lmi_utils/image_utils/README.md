## Documentation

*work in progress*

This package contains a set of utilities for image processing and computer vision. The package is divided into several modules, each of which contains a set of functions for a specific task. The modules are:

- `allignment_utils.py`: This file contains functions to get the contours of the image, get the bounding box of the contours, and align the image based on the bounding box.

#### Functions

- `getContours`: This function takes an image as input and returns the contours of the image.
- `maskByContour`: This function takes an image and a contour as input and returns the masked image.
- `rotateByContour`: This function takes an image and a contour as input and returns the rotated image based on the contour.
- `rotateByContour`: This function takes an image and a contour as input and returns the rotated image based on the contour.
- `getContCenter`: This function takes a contour as input and returns the center of the contour.
- `rotateByMinBB`: This function takes an image and a contour as input and returns the rotated image based on the minimum bounding box of the contour.
- `cropByBB`: This function takes an image and a contour and returns the cropped image based on the bounding box of the contour.
- `extract_UniformBox_ROI_from_JSON`: This function takes a json_file_path, input_image_dir_path, and output_image_dir_path as input and saves the cropped images based on the bounding box of the contour in the json file.
- `align_and_crop`: This function takes an image_file_path and a output_dir_path as input and performs allignment, cropping based on the bounding box and saves the image in the provided output directory.
- `tile_image`: This function takes an image_file_path, tile_dir_path, window_dimension, steps_per_window, performs tiling of the image and saves the tiles in the tile_dir_path.
