# conftest.py
import logging
import os
import sys

# ultralytics AutoUpdate pip-installs CPU onnxruntime over onnxruntime-gpu, silently killing the CUDA provider.
# Must be set before anything imports ultralytics, which reads this at import time.
os.environ.setdefault("YOLO_AUTOINSTALL", "false")

logging.basicConfig()


def pytest_configure(config):
    # Set up logging specifically for pytest_configure
    logger = logging.getLogger(__name__)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    # Prevent logging from propagating to the root logger
    logger.propagate = False
