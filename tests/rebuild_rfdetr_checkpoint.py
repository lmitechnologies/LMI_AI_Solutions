"""Re-stamp the committed RF-DETR test checkpoint with the variant name rfdetr needs to identify it.

``RfdetrPTH`` reads the variant from the checkpoint (``RFDETR.from_checkpoint``) instead of taking a
``model_type`` argument. rfdetr writes that ``model_name`` key from 1.7.0 on, but Roboflow's published
starter weights predate it and hold nothing but a ``model`` state dict, so ``from_checkpoint`` fails on
them. This script downloads the starter weights and rewrites them with the keys rfdetr looks for, so the
test asset exercises the same path a user's own trained checkpoint takes.

    python -m tests.rebuild_rfdetr_checkpoint            # overwrite the committed asset
    python -m tests.rebuild_rfdetr_checkpoint -o /tmp/x.pth

The weights are untouched — only metadata is added. Needs ``rfdetr``; runs on CPU.
"""

import argparse
import logging
import os
import urllib.request

import torch

logger = logging.getLogger("rebuild_rfdetr_checkpoint")

OUT_PATH = "tests/assets/models/od/rf_detr/rf-detr-seg-small.pth"
# rfdetr's own RF_DETR_SEG_SMALL asset entry (rfdetr/assets/model_weights.py).
SOURCE_URL = "https://storage.googleapis.com/rfdetr/rf-detr-seg-s-ft.pth"
SOURCE_MD5 = "0a2a3006381d0c42853907e700eadd08"
MODEL_NAME = "RFDETRSegSmall"


def download(url: str, dest: str) -> None:
    """Fetch url to dest unless dest already exists."""
    if os.path.isfile(dest):
        logger.info("reusing %s", dest)
        return
    logger.info("downloading %s ...", url)
    urllib.request.urlretrieve(url, dest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", default=OUT_PATH, help=f"where to write the checkpoint (default: {OUT_PATH})")
    parser.add_argument("--source", default=None, help="starter weights to re-stamp; downloaded from Roboflow when omitted")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    source = args.source
    if source is None:
        source = os.path.join("/tmp", os.path.basename(SOURCE_URL))
        download(SOURCE_URL, source)

    from rfdetr import RFDETRSegSmall

    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    if "model" not in checkpoint:
        raise ValueError(f"{source} has no 'model' state dict; keys: {sorted(checkpoint)}")

    # Build the variant off the same weights so its resolved config is what gets recorded.
    model_config = RFDETRSegSmall(pretrain_weights=source, device="cpu").model_config.model_dump()
    model_config.pop("pretrain_weights", None)  # from_checkpoint sets this to the checkpoint's own path
    model_config["model_name"] = MODEL_NAME

    # "args" and "model_name" are what from_checkpoint reads; "model_config" seeds the constructor kwargs.
    payload = {"model": checkpoint["model"], "args": model_config, "model_name": MODEL_NAME, "model_config": model_config}
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save(payload, args.output)
    logger.info("wrote %s (%.1f MB)", args.output, os.path.getsize(args.output) / 1e6)


if __name__ == "__main__":
    main()
