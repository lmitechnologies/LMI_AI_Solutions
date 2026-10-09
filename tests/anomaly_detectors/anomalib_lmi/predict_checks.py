import os

import cv2
import numpy as np

from lmi_utils.preprocess_utils import steps
from lmi_utils.preprocess_utils.preprocessor import Preprocessor
from lmi_utils.preprocess_utils.reconstructor import Reconstructor

IMAGE_PATH = os.path.join("tests/assets/images/nvtec-ad", "000-bad.png")


def assert_predict_operators_match_reconstructor(model, batch_size=None):
    """``predict(tiles, operators=history)`` must equal reverting the plain ``predict(tiles)`` maps with a Reconstructor."""
    image = cv2.cvtColor(cv2.imread(IMAGE_PATH), cv2.COLOR_BGR2RGB)
    h, w = model.image_size
    tiles, history = Preprocessor().preprocess([image], [steps.tile(tile_size=[h, w], stride=[h - 24, w - 24])])
    direct = model.predict(tiles, operators=history, batch_size=batch_size)
    reverted = Reconstructor().reconstruct_images(model.predict(tiles, batch_size=batch_size), history)
    assert len(direct) == 1 and direct[0].shape[:2] == image.shape[:2]
    np.testing.assert_array_equal(direct[0], reverted[0])
