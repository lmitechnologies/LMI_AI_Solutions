import argparse
import glob
import os

import open3d

from lmi_utils.pcl_utils.point_cloud import PointCloud


def convert_npy_to_pcd(path_in, path_out):
    files = glob.glob(os.path.join(path_in, "*.npy"))

    pointcloud = PointCloud()
    for file in files:
        pointcloud.read_points(file, zmin=None, zmax=None, clip_mode=False)
        pcd = pointcloud.get_PCD()
        file_out = os.path.split(file)[1].replace(".npy", ".pcd")
        print(f"[INFO] .pcd output file: {file_out}")
        file_out = os.path.join(path_out, file_out)
        open3d.io.write_point_cloud(file_out, pcd)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input_path", default="./")
    ap.add_argument("-o", "--output_path", default="./")
    args = vars(ap.parse_args())
    path_in = args["input_path"]
    path_out = args["output_path"]

    convert_npy_to_pcd(path_in, path_out)
