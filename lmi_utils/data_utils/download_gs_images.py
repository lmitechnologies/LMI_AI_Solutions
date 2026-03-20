import argparse
import json
import logging
import os
import subprocess

logger = logging.getLogger(__name__)


def download_gs_images(input_file, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. Extract URIs from JSON
    gs_uris = []
    try:
        with open(input_file, "r") as f:
            data = json.load(f)
            # Handle both a single object or a list of objects
            tasks = data if isinstance(data, list) else [data]

            for task in tasks:
                uri = task.get("data", {}).get("image")
                if uri and uri.startswith("gs://"):
                    gs_uris.append(uri)
    except FileNotFoundError:
        logger.error(f"{input_file} not found.")
        return

    if not gs_uris:
        logger.info("No Google Storage (gs://) URIs found.")
        return

    # 2. Write URIs to a temporary file for gsutil to read
    manifest_path = "gs_uris_list.txt"
    with open(manifest_path, "w") as f:
        for uri in gs_uris:
            f.write(f"{uri}\n")

    logger.info(f"Found {len(gs_uris)} images. Starting parallel download...")

    # 3. Use gsutil -m (multithreading) to download in parallel
    try:
        with open(manifest_path, "r") as manifest_file:
            subprocess.run(
                ["gsutil", "-m", "cp", "-n", "-I", output_dir],
                stdin=manifest_file,
                shell=True,
                check=True,
            )
        logger.info(f"\nSuccess! Images saved to {output_dir}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error during gsutil execution: {e}")
    finally:
        # Clean up temporary manifest
        if os.path.exists(manifest_path):
            os.remove(manifest_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Download images from Google Storage URIs in a label studio JSON file.")
    parser.add_argument("-j", "--json", required=True, help="Path to the label studio JSON file.")
    parser.add_argument("-o", "--output", required=True, help="Output directory for downloaded images.")
    args = parser.parse_args()

    download_gs_images(args.json, args.output)
