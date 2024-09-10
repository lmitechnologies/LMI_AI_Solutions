import abc
import gadget_utils.pipeline_utils as pipeline_utils



class ModelBase(abc.ABC):

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
    