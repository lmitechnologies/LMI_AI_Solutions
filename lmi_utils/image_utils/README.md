## Documentation

*@Rugved remove all TODO's and comments before merge*

#### Scripts

- `allignment_utils.py`: This file contains functions to get the contours of the image, get the bounding box of the contours, and align the image based on the bounding box.
- `flip_image.py`: This script flips the image either horizontally or vertically given a path to the folder containing the images and saves them in a given folder. Arguments for the script: `--path_imgs [INPUT_FOLDER_PATH] --path_out [OUTPUT_FOLDER_PATH] --flip ["ud"(vertical) or "lr" (horizontal)]
- `image_pad_to_size.py`:  This script pads an image or set of images to a given height and width. Arguments for the script: --input_image_path [INPUT_IMAGE_PATH or INPUT_IMAGES_PATH] --output_path [OUTPUT_IMAGE_PATH or OUTPUT_FOLDER_PATH] --height [PAD_HEIGHT] --width [PAD_WIDTH]
- `image_resize.py`: This script resizes an image or set of images to a given height and width. Arguments for the script: --input_path [INPUT_IMAGE_PATH or INPUT_FOLDER_PATH] --output_path [OUTPUT_IMAGE_PATH or OUTPUT_IMAGES_PATH] --height [RESIZE_HEIGHT] --width [RESIZE_WIDTH]
- `image_rotate.py`: This file contains functions to rotate an image given the image and angle.
- `image_stretch.py`: This file perfoms a resize operation to a given image or set of images provided scale. Arguments for the script: --input_path [INPUT_IMAGE_PATH or INPUT_FOLDER_PATH] --output_path [OUTPUT_IMAGE_PATH or OUTPUT_FOLDER_PATH] --scale [STRETCH_SCALE]
- `make_collage.py`: This script is used to create a collage given a set of images by stacking them horizontally. Arguments for the script: --input_data_path [INPUT_IMAGE_PATH or INPUT_FOLDER_PATH] --output_image_path [OUTPUT_FOLDER_PATH] --width [COLLAGE_WIDTH] --height [COLLAGE_HEIGHT] --max_rows_per_collage [MAX_ROWS_PER_COLLAGE]
- `merge_pred_with_orig.py`: This script is used to perform an horizontal stacking of the original image and the predicted image. Arguments for the script: --path_orig [INPUT_FOLDER_PATH_ORIGINAL] --path_pred [INPUT_FOLDER_PATH_PREDICTED] --path_out [OUTPUT_FOLDER_PATH] --fmt ["png" or "jpg"]
- `numpy_utils.py`: This script performs either of the three options on an image npy_2_png(numpy file to png) or png_2_npy(png file to numpy file) or png_2_png (converts an image from rgb to png). Arguments for the script: --src [INPUT_FOLDER_PATH] --dest [OUTPUT_FOLDER_PATH] --option ["npy_2_png" or "png_2_npy" or "png_2_png"]
- `pad_image.py`: This script pads an image or set of images to a given height and width. Arguments for the script: --path_imgs [INPUT_FOLDER_PATH] --path_out [OUTPUT_FOLDER_PATH] --out_imsz [[PAD_WIDTH, PAD_HEIGHT]] --keep_same_filename [True or False]
- `resize_image.py`: This script resizes an image or set of images to a given height and width, while maintaining aspect ratio. Arguments for the script: --path_imgs [INPUT_FOLDER_PATH] --path_out [OUTPUT_FOLDER_PATH] --out_imsz [[RESIZE_WIDTH, RESIZE_HEIGHT]] --keep_same_filename [True or False]
- `rgb_convert.py`: This script converts an image or set of images from greyscale to rgb. Arguments for the script: None
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

###### flip_image.py

- `flip_images`: This function takes a list of image paths, flip type (ud, lr), and output directory path as input and saves the flipped images in the output directory.

###### img_pad_to_size.py

- `pad_image`: This function takes an image_file_path or image folder path, output_path, pad width, pad height, and saves the padded image in the provided output path.

- `pad_array`: This function takes an image, pad width, pad height, and returns the padded image.

###### img_resize.py

- `is_cuda_cv`: This function checks whether the CUDA version of OpenCV is in use and returns True if it is, False otherwise.

- `resize`: This function takes an image, resize width, resize height, device (optional 'cpu' or 'cuda') and returns the resized image.

###### image_rotate.py

- `rotate`: This function takes an image, angle, center, and scale as parameters and returns the rotated image. Performs rotation of the image by determing the rotation matrix and applying the matrix to the image.

###### image_stretch.py

- `stretch`: This function takes an image and scale as input and returns the stretched image (applies scale based on which one is the biggest width or height).

- `gen_collage`: This function takes the following inputs input_path (folder of images), output_path (location to store the collage image), colmax (max number of columns), width (resize width before stacking), rowmax (max number of rows), file_filter (file name filter as a string). It saves the colleges in the given output path.

###### merge_pred_with_orig.py

- `hstack_imgs`: This function takes folder of images, folder of predictions results, image format (fmt), and output path to store the merged image. Performs an horizontal stacking of the original image and the predicted image and saves it in the output path.

###### numpy_utils.py

*The following are methods of the class NumpyUtils*

- `png_to_npy`: This function takes an image folder path, folder to save the images, rotate (rotate the image by 90 deg) and rgb2bgr (boolean to convert the image from RBG to BGR). It saves the images as .npy files in the provided folder path.

- `npy_to_png`: This function takes an image folder path, folder to save the images, rotate (rotate the image by 90 deg) and rgb2bgr (boolean to convert the image from RBG to BGR). It reads in all the .npy files in the provided folder path and saves them as .png files in the provided folder path.

- `png_to_png`: This function takes an image folder path, folder to save the images, rotate (rotate the image by 90 deg) and rgb2bgr (boolean to convert the image from RBG to BGR). It reads in all the .png files in the provided folder path and saves them as .png files in the output folder.

###### pad_image.py

- `fit_image_to_size`: This function takes a folder of images and an output path to store padded/cropped images, output image size, and whether to use the same filename (boolean to keep the same filename). It pads the images to the provided size and saves them in the output path.

###### resize_image.py

- `resize_images`: *maintains aspect ratio* This function takes a folder of images and an output path to store resized images, output image size, and whether to use the same filename (boolean to keep the same filename). It pads the images to the provided size and saves them in the output path.

###### rgb_convert.py

- `convert_to_rainbow`: This function converts an integer to full scale range based on the precision chosen. It converts the integer to the rgb space and returns the values.

- `convert_array_to_rainbow`: This function converts an array of integers to full scale range based on the precision chosen. It converts the integers to the rgb space and returns the values.

- `convert_from_rgb`: This function converts an rgb value to an integer. It converts the rgb value to an integer and returns the value.

- `convert_greyscale_image_to_color`: This function takes a greyscale image and converts it to a color image. It converts the greyscale image to a color image and returns the image.

- `convert_greyscale_to_color_simple` : This function converts the greyscale image by expaning the image to 3 channels. It converts the greyscale image to a color image and returns the image.

###### sample_image.py

- `sample_images`: This function takes a folder of images, output path to store sampled images, number of samples to be taken, and whether to sample randomly (boolean to sample randomly). It samples the images and saves them in the output path.

###### split_stack_image.py

- `split_stack_image`: This function a folder of images, output path to store split images, number of splits, stack type (horizontal or vertical), and whether to use the same filename (boolean to keep the same filename). It splits the images and stacks them based on the provided type and saves them in the output path.

- `split_hstack_image`: This function takes an image, number of splits (vertically), and returns the stacked image. It horizontally stacks them and returns the stacked image.

- `split_vstack_image`: This function takes an image, number of splits (horizontally), and returns the stacked image. It vertically stacks them and returns the stacked image.

###### tile_image.py

- `__tile_image`: This function takes an image path, output path, tile width, tile height, stride width, stride height, overlap mode and saves the tiled images. It tiles the image and saves the tiles in the output path.

- `tile_image`: This function takes an image path, output path, tile width, tile height, stride width, stride height, overlap mode and saves the tiled images. It tiles the image and saves the tiles in the output path. (achieved by calling __tile_image)

- `reconstruct_image`: This function takes in the folder with tiled images, output path, overlap mode (avg or max) and reconstructs the image, following saves the reconstructed image in the given destination folder.
