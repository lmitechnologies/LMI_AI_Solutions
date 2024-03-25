## Documentation

*work in progress*

This package contains a set of utilities for image processing and computer vision. The package is divided into several modules, each of which contains a set of functions for a specific task. The modules are:

- `allignment_utils.py`: This file contains functions to get the contours of the image, get the bounding box of the contours, and align the image based on the bounding box.
- `flip_image.py`: This script flips the image either horizontally or vertically given a path to the folder containing the images and saves them in a given folder. Arguments for the script: `--path_imgs [INPUT_FOLDER_PATH] --path_out [OUTPUT_FOLDER_PATH] --flip ["ud"(vertical) or "lr" (horizontal)]
- `image_pad_to_size.py`:  This script pads an image or set of images to a given height and width. Arguments for the script: --input_image_path [INPUT_IMAGE_PATH or INPUT_IMAGES_PATH] --output_path [OUTPUT_IMAGE_PATH or OUTPUT_FOLDER_PATH] --height [PAD_HEIGHT] --width [PAD_WIDTH]
- `image_resize.py`: This script resizes an image or set of images to a given height and width. Arguments for the script: --input_path [INPUT_IMAGE_PATH or INPUT_FOLDER_PATH] --output_path [OUTPUT_IMAGE_PATH or OUTPUT_IMAGES_PATH] --height [RESIZE_HEIGHT] --width [RESIZE_WIDTH]
- `image_rotate.py`: This file contains functions to rotate an image given the image and angle.
- `image_stretch.py`: *can be achieved by image_resize.py?* This file perfoms a resize operation to a given image or set of images provided scale. Arguments for the script: --input_path [INPUT_IMAGE_PATH or INPUT_FOLDER_PATH] --output_path [OUTPUT_IMAGE_PATH or OUTPUT_FOLDER_PATH] --scale [STRETCH_SCALE]
- `make_collage.py`: This script is used to create a collage given a set of images by stacking them horizontally. Arguments for the script: --input_data_path [INPUT_IMAGE_PATH or INPUT_FOLDER_PATH] --output_image_path [OUTPUT_FOLDER_PATH] --width [COLLAGE_WIDTH] --height [COLLAGE_HEIGHT] --max_rows_per_collage [MAX_ROWS_PER_COLLAGE]
- `mege_pred_with_orig.py`: This script is used to perform an horizontal stacking of the original image and the predicted image. Arguments for the script: --path_orig [INPUT_FOLDER_PATH_ORIGINAL] --path_pred [INPUT_FOLDER_PATH_PREDICTED] --path_out [OUTPUT_FOLDER_PATH] --fmt ["png" or "jpg"]
- `numpy_utils.py`: This script performs either of the three options on an image npy_2_png(numpy file to png) or png_2_npy(png file to numpy file) or png_2_png (converts an image from rgb to png). Arguments for the script: --src [INPUT_FOLDER_PATH] --dest [OUTPUT_FOLDER_PATH] --option ["npy_2_png" or "png_2_npy" or "png_2_png"]
- `pad_image.py`: This script pads an image or set of images to a given height and width. Arguments for the script: --path_imgs [INPUT_FOLDER_PATH] --path_out [OUTPUT_FOLDER_PATH] --out_imsz [[PAD_WIDTH, PAD_HEIGHT]] --keep_same_filename [True or False]
- `resize_image.py`: This script resizes an image or set of images to a given height and width, while maintaining aspect ratio. Arguments for the script: --path_imgs [INPUT_FOLDER_PATH] --path_out [OUTPUT_FOLDER_PATH] --out_imsz [[RESIZE_WIDTH, RESIZE_HEIGHT]] --keep_same_filename [True or False]
- `rgb_convert.py`: This script converts an image or set of images from greyscale to rgb. Arguments for the script: None *TODO: ask Chris regarding this script*
- `sample_image.py`: This script samples a number of images from a given list of images and copies them into a folder. Arguments for the script: --path_imgs [INPUT_FOLDER_PATH] --path_out [OUTPUT_FOLDER_PATH] --num_samples [NUMBER_OF_SAMPLES] --random [True or False]
- `split_stack_image.py`: This script splits images horizontally or vertically and stacks them on the choice. Arguments for the script: --path_imgs [INPUT_FOLDER_PATH] --path_out [OUTPUT_FOLDER_PATH] --num_splits [NUMBER_OF_SPLITS] --stack ["h" or "v"] --keep_same_filename [True or False]
- `tiff_2_png.py`: This script applies a color map to a provided tiff image and save it as a .png file. Arguments for the script: --input_path [INPUT_FOLDER_PATH] --output_path [OUTPUT_FOLDER_PATH] --cvt_hmap_jet [True or False] (applies a jet color map to the image)
- `tile_image.py`: This script tiles an image or set of images, moreover also provides an option to reconstruct tiles to a single image. Arguments for the script: --option ["tile" or "reconstruct"] --src [INPUT_FOLDER_PATH] --dest [OUTPUT_FOLDER_PATH] --w [TILE_WIDTH] --h [TILE_HEIGHT] --stride_w [TILE_STRIDE_WIDTH] --stride_h [TILE_STRIDE_HEIGHT] --overlap_mode ["avg" or "max"] --type [IMAGE_FORMAT]

#### Functions

##### allignment_utils.py

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

###### image_rotate.py

- `rotate`: This function takes an image, angle, center, and scale as parameters and returns the rotated image. Performs rotation of the image by determing the rotation matrix and applying the matrix to the image.
