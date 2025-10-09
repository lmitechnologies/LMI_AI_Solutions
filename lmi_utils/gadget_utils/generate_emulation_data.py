from pathlib import Path
import pickle
import numpy as np
import cv2
import tempfile
import json
import tarfile


SCHEMA_ID: str = "gadget3d"
VERSION: int = 1


def to_int16(arr):
    """
    Convert an array to int16, handling different data types.
    """
    if arr.dtype == np.uint16:
        return arr.view(np.int16) + np.int16(-32768)
    elif arr.dtype == np.int32:
        return (arr - 32768).astype(np.int16)
    elif arr.dtype == np.int16:
        return arr
    else:
        raise ValueError(f"Unsupported data type: {arr.dtype}")


def generate_emulation_data(path_source, path_out):
    path_source = Path(path_source)
    path_out = Path(path_out)

    if not path_out.exists():
        path_out.mkdir(parents=True)
        
    # look for each of image_* folders and corresponding surface_* folders
    image_folders = sorted(path_source.rglob("image_*"))
    surface_folders = sorted(path_source.rglob("surface_*"))
    
    print(f"Found {len(image_folders)} image folders and {len(surface_folders)} surface folders.")

    for image_folder, surface_folder in zip(image_folders, surface_folders):
        print(f"Processing {image_folder.name} and {surface_folder.name}")
        image_files = sorted(image_folder.glob("*.jpg"))
        surface_files = sorted(surface_folder.glob("*.tar"))
        
        for image_file, surface_file in zip(image_files, surface_files):
            print(f"Processing image file {image_file.name} and surface file {surface_file.name}")
            img = cv2.imread(str(image_file), cv2.IMREAD_UNCHANGED)
            
            # laod surface data
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_path = Path(tmpdir)
                # extract surface tar file
                with tarfile.open(surface_file, "r") as tar:
                    tar.extractall(path=tmp_path)
                
                # load json metadata
                metadata_file = tmp_path / "metadata.json"
                if not metadata_file.exists():
                    raise FileNotFoundError(f"Metadata file not found in {tmp_path}.")
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                resolution = metadata.get("resolution")
                offset = metadata.get("offset")
                    
                # load surface data
                surface_data_file = tmp_path / "profile.png"
                if not surface_data_file.exists():
                    raise FileNotFoundError(f"Surface data file not found in {tmp_path}.")
                surface_data = cv2.imread(str(surface_data_file), cv2.IMREAD_UNCHANGED)
                surface_data = to_int16(surface_data)

            content = {
                "metadata": {
                    "schema": SCHEMA_ID,
                    "version": VERSION,
                    "resolution": resolution,
                    "offset": offset,
                },
                "profile_array": surface_data,
                "intensity_array": img,
            }

            out_file = path_out / f"{image_file.stem}.gadget3d.pickle"
            with open(out_file, "wb") as f:
                pickle.dump(content, f, protocol=4)
                
    
    
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate emulation data from source images.")
    parser.add_argument("--path_source", "-i", type=str, help="Path to the source directory containing image_* and surface_* folders.")
    parser.add_argument("--path_out", "-o", type=str, help="Path to the output directory where the gadget3d files will be saved.")

    args = parser.parse_args()

    generate_emulation_data(args.path_source, args.path_out)
    