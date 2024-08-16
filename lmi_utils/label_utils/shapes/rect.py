from .shape import Shape

class Rect(Shape):
    """
    the rectangle class for bounding box annotations
    """
    def __init__(self, im_name='', fullpath='', category='', up_left=[0,0], bottom_right=[0,0], confidence=1.0, angle=0):
        """
        Arguments:
            im_name(str): the image file basename
            fullpath(str): the location of the image file
            category(str): the categorical class name of this image
            up_left(list): the up left point [x1,y1] of the bbox
            bottom_right(list): the bottom right point [x2,y2] of the bbox
            confidence(double): the confidence level between [0.0, 1.0] 
        """
        super().__init__(im_name,fullpath,category,confidence)
        self.up_left = up_left
        self.bottom_right = bottom_right
        self.angle = angle
        
    def round(self):
        self.up_left = list(map(round,self.up_left))
        self.bottom_right = list(map(round,self.bottom_right))
        