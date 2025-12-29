import numpy as np
from image_utils.img_resize import resize_and_pad


class Preprocessor:
    """
    A class to run a dynamic pipeline of preprocessing steps on an image.

    Handlers (processing functions) are registered with the instance and
    called based on a list of processing steps.
    """

    def __init__(self):
        """
        Initializes the preprocessor and registers default handlers.
        """
        self._handlers = {}
        self._register_default_handlers()

    def _register_default_handlers(self):
        """Registers the built-in processing functions."""
        self.register_handler("resize", resize_and_pad)

    def register_handler(self, name: str, handler_func):
        """
        Registers a new handler function or overwrites an existing one.

        The handler_func must accept 'image' as its first argument,
        followed by keyword arguments matching its configuration.
        """
        if not callable(handler_func):
            raise TypeError(f"Handler for '{name}' must be a callable function.")
        self._handlers[name] = handler_func

    def preprocess(self, image: np.ndarray, processing_steps: list) -> np.ndarray:
        """
        Applies a sequence of preprocessing steps to the input image.

        Args:
            image (np.ndarray): The input image to preprocess.
            processing_steps (list): A list of dictionaries, each specifying
                                     a processing step with its parameters.

        Returns:
            np.ndarray: The preprocessed image.
        """
        im_out = image
        operators = []
        for step in processing_steps:
            op_name = step.get("type")
            config = step.get("configuration", {})
            if op_name not in self._handlers:
                raise ValueError(f"Handler for '{op_name}' is not registered.")
            if not isinstance(config, dict):
                raise TypeError(f"Configuration for '{op_name}' must be a dictionary.")
            if config is None or config == {}:
                raise ValueError(f"Configuration for '{op_name}' cannot be None or empty.")

            handler_func = self._handlers[op_name]
            # Pass the current image and step parameters to the handler
            im_out, operators = handler_func(im_out, **config, operators=operators, return_operators=True)

        return im_out, operators
