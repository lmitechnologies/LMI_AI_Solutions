from abc import ABC, abstractmethod

import numpy

from fringe_ds_utils.geometry.primitives import Box, Patch


class ImageOrPointCloud(ABC):
    def __init__(self):
        self.name = None
        self.img = None
        self.TWO_TO_TWENTYFOURTH_MINUS_ONE = 16777215
        super().__init__()

    @abstractmethod
    def read_points(self, path, mode="open_3d"):
        pass

    @abstractmethod
    def extract_patches_from_img(self, m, s):
        pass

    def extract_patches_from_img(self, m, s):
        """
        Notation from DOI: 10.1109/TCYB.2017.2668395
        :param m: patch size
        :param s: stride size
        :return: generator of patches
        """
        M, N = numpy.shape(self.img)[0], numpy.shape(self.img)[1]
        boxes = (
            Box(x_start, x_start + m, y_start, y_start + m)
            for x_start in range(0, N - m + 1, s)
            for y_start in range(0, M - m + 1, s)
        )
        return (
            Patch(box, self.img[box.ymin : box.ymax, box.xmin : box.xmax])
            for box in boxes
        )
