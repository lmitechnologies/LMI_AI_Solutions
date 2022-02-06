from shape import Shape

class Rect(Shape):
    """
    the rectangle class for bounding box annotations
    """
    def __init__(self, im_name='', fullpath='', category='', up_left=[0,0], bottom_right=[0,0]) -> None:
        """
        Arguments:
            im_name(str): the image file basename
            fullpath(str): the location of the image file
            category(str): the categorical class name of this image
            up_left(list): the up left point [x1,y1] of the bbox
            bottom_right(list): the bottom right point [x2,y2] of the bbox
        """
        super(Rect, self).__init__(im_name,fullpath,category)
        self.up_left = up_left
        self.bottom_right = bottom_right
