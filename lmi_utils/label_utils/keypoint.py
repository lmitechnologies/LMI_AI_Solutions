from label_utils.shape import Shape

class Keypoint(Shape):
    def __init__(self, im_name='', fullpath='', category='', x=0.0, y=0.0, confidence=1.0):
        super().__init__(im_name, fullpath, category, confidence)
        self.x = x
        self.y = y
        
    def round(self):
        self.x = round(self.x)
        self.y = round(self.y)
        