import logging

import numpy as np

from .shape import Shape

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


class Brush(Shape):
    """
    the class for polygon brush annotations
    """

    def __init__(self, im_name="", fullpath="", category="", mask=None, confidence=1.0):
        """
        Arguments:
            im_name(str): the image file basename
            fullpath(str): the location of the image file
            category(str): the categorical class name of this image
            mask(np.ndarray): the binary mask of a decoded rle image
            confidence(double): the confidence level between [0.0, 1.0]
        """
        super().__init__(im_name, fullpath, category, confidence)
        self.X = []
        self.Y = []
        if mask is not None:
            # np.where returns (row,col), whereas opencv (x,y) is (col,row).
            self.Y, self.X = np.where(mask)

    def round(self):
        self.X = list(map(round, self.X))
        self.Y = list(map(round, self.Y))

    def to_mask(self, hw):
        """convert to a binary mask

        Args:
            hw (list): a list of h and w

        Returns:
            np.ndarray: a binary mask
        """
        mask = np.zeros(hw, dtype=bool)
        self.round()
        mask[self.Y, self.X] = True
        return mask
