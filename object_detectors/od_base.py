import abc

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
    