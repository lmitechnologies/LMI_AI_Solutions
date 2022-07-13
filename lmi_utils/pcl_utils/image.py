import os

import cv2
import numpy

from fringe_ds_utils.pcl_utils.image_or_point_cloud import ImageOrPointCloud


class Image(ImageOrPointCloud):
    def __init__(self):
        super().__init__()

    def read_points(self, path, mode="open_3d"):
        self.name = os.path.splitext(os.path.split(path)[-1])[0]
        self.img = cv2.imread(path)
