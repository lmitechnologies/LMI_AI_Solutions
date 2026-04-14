# Detectron2TRT

The `Detectron2TRT` class provides an interface for using a Detectron2 model with TensorRT, allowing for efficient and optimized inference on NVIDIA GPUs. This document covers the features of the `Detectron2TRT` class, how to initialize it, and how to use its various methods to perform object detection tasks.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
  - [Initialization](#initialization)
  - [Warmup](#warmup)
  - [Preprocess](#preprocess)
  - [Forward Pass](#forward-pass)
  - [Postprocess](#postprocess)
  - [Predict](#predict)
  - [Annotate Image](#annotate-image)

## Installation

Make sure you have the required libraries installed, including TensorRT, PyCUDA, Detectron2, and other dependencies.

```sh
pip install tensorrt pycuda numpy torch opencv-python
```
## Usage

### Initialization

To initialize the `Detectron2TRT` model, provide the path to the serialized TensorRT engine file and a dictionary that maps class IDs to class names.

```python
from detectron2_lmi.model import Detectron2TRT

model_path = "path/to/your/model.engine"
class_map = {0: "background", 1: "person", 2: "car"}
model = Detectron2TRT(model_path, class_map)
```

### Warmup

The `warmup` method runs a forward pass with a randomly generated input tensor to reduce initial inference latency.

**Description:**

Perform a warmup operation for the model.

This method runs a forward pass with a randomly generated input tensor to warm up the model. It helps in preparing the model for actual inference by initializing necessary components and reducing the initial latency.

**Usage Example:**

```python
model.warmup(img_size=[640,640]) # img_size H,W
```

### Preprocess

The `preprocess` method converts a batch of images into the correct format for the model. The images should be provided in the format `(H, W, C)`.

**Description:**

Preprocesses a batch of images for input into the model.

**Args:**
- `images (np.ndarray | list)`: A batch of images to preprocess. Each image should be in the format (H, W, C).

**Returns:**
- `np.ndarray`: A batch of preprocessed images with shape (batch_size, 3, image_h, image_w).

**Usage Example:**

```python
import numpy as np

# Sample batch of images (list of np.ndarray in the format HxWxC)
images = [np.random.rand(640, 640, 3).astype(np.float32) for _ in range(model.batch_size)]
preprocessed_images = model.preprocess(images)
```

### Forward Pass

The `forward` method runs a forward pass through the model using the preprocessed inputs.

**Description:**

Perform a forward pass through the model.

**Args:**
- `inputs (numpy.ndarray)`: The input data to be processed by the model.

**Returns:**
- `list`: A list of numpy arrays containing the model's output data.

**Usage Example:**

```python
# Running forward pass
outputs = model.forward(preprocessed_images)
```

### Postprocess

The `postprocess` method processes the model's output and extracts information like bounding boxes, class labels, scores, and masks.

**Description:**

Post-process the predictions from the object detection model.

**Args:**
- `images (list)`: List of input images.
- `predictions (tuple)`: Tuple containing the number of predictions, bounding boxes, scores, classes, and masks.
- `**kwargs`: Additional keyword arguments for processing.
  - `confs (dict)`: Dictionary of confidence thresholds for each class.
  - `mask_threshold (float)`: Threshold for mask binarization.
  - `process_masks (bool)`: Flag to indicate whether to process masks.

**Returns:**
- `dict`: A dictionary containing the processed results with keys: "boxes", "scores", "classes", "masks".

**Usage Example:**

```python
predictions = model.postprocess(images, outputs, confs={"person": 0.5})
```

### Predict

The `predict` method combines the preprocessing, forward pass, and postprocessing steps into one. It outputs the final predictions, including bounding boxes, scores, and class labels.

**Description:**

Perform prediction on the given images.

**Args:**
- `images (list or np.ndarray)`: The input images to be processed.
- `**kwargs`: Additional keyword arguments for postprocessing.

**Returns:**
- `list`: The predictions after postprocessing.

**Logs:**
- The time taken for postprocessing in milliseconds.

**Usage Example:**

```python
# Running prediction
results = model.predict(images, confs={"person": 0.5})
```

### Annotate Image

The `annotate_image` method can be used to annotate the input image with bounding boxes, class labels, and masks. Optionally, you can provide a color map to specify different colors for each class.

**Description:**

Annotates an image with bounding boxes, class labels, and masks.

**Args:**
- `result (dict)`: A dictionary containing detection results with keys: "classes", "boxes", "scores", "masks".
- `image (numpy.ndarray)`: The image to annotate.
- `color_map (dict, optional)`: A dictionary mapping class labels to colors.

**Returns:**
- `numpy.ndarray`: The annotated image.

**Usage Example:**

```python
import cv2

# Annotate the image
annotated_image = model.annotate_image(result, images[0])

# Display the image
cv2.imshow("Annotated Image", annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

## Acknowledgments

This project is based on NVIDIA's TensorRT and Detectron2. For more information, check out the [Detectron2 GitHub repository](https://github.com/facebookresearch/detectron2) and [TensorRT GitHub repository](https://github.com/NVIDIA/TensorRT).