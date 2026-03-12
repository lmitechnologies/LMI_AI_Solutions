import abc


class ClassifierBase(abc.ABC):
    @abc.abstractmethod
    def warmup(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def preprocess(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def postprocess(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def predict(self, *args, **kwargs):
        """
        combine preprocess, forward, and postprocess
        """
        pass
