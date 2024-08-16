from .shape import Shape

class Mask(Shape):
    """
    the class for polygon mask annotations
    """
    def __init__(self, im_name='', fullpath='', category='', x_vals=[], y_vals=[], confidence=1.0):
        """
        Arguments:
            im_name(str): the image file basename
            fullpath(str): the location of the image file
            category(str): the categorical class name of this image
            x_vals(list): the x values [x1, x2, ..., xn] of the polygon
            y_vals(list): the y values [y1, y2, ..., yn] of the polygon
            confidence(double): the confidence level between [0.0, 1.0] 
        """
        super().__init__(im_name,fullpath,category,confidence)
        self.X = x_vals
        self.Y = y_vals

    def round(self):
        self.X = list(map(round,self.X))
        self.Y = list(map(round,self.Y))
        