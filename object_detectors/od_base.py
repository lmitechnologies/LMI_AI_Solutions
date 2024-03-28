import abc
import gadget_utils.pipeline_utils as pipeline_utils



class ODBase(abc.ABC):

    def __init__(self):
        pass
    
    
    @abc.abstractmethod
    def warmup(self):
        pass
    
    
    @abc.abstractmethod
    def preprocess(self):
        pass


    @abc.abstractmethod
    def forward(self):
        pass


    @abc.abstractmethod
    def postprocess(self):
        pass
    
    
    @abc.abstractmethod
    def predict(self):
        """
        combine preprocess, forward, and postprocess
        """
        pass
    
    
    @staticmethod
    def annotate_image(results, image, colormap=None):
        """annotate the object dectector results on the image. If colormap is None, it will use the random colors.
        TODO: text size, thickness, font

        Args:
            results (dict): the results of the object detection, e.g., {'boxes':[], 'classes':[], 'scores':[], 'masks':[], 'segments':[]}
            image (np.ndarray): the input image
            colors (list, optional): a dictionary of colormaps, e.g., {'class-A':(0,0,255), 'class-B':(0,255,0)}. Defaults to None.

        Returns:
            np.ndarray: the annotated image
        """
        boxes = results['boxes']
        classes = results['classes']
        scores = results['scores']
        masks = results['masks']
        
        image2 = image.copy()
        if not len(boxes):
            return image2
        
        for i in range(len(boxes)):
            mask = masks[i] if len(masks) else None
            pipeline_utils.plot_one_box(
                boxes[i],
                image2,
                mask,
                label="{}: {:.2f}".format(
                    classes[i], scores[i]
                ),
                color=colormap[classes[i]] if colormap is not None else None,
            )
        return image2