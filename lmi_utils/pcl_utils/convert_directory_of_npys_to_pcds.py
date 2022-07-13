import numpy
import open3d
import time

import glob
import os
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
                    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class DirectoryConverter_npy_to_pcd(object):
    def __init__(self, input_directory, output_directory):
        self.input_directory = input_directory
        self.output_directory = output_directory

    def convert(self, reverse=False):
        if not os.path.exists(self.output_directory):
            os.mkdir(self.output_directory)
            logger.info(f"Created {self.output_directory}.")
        convert_function = convert_file_pcd_to_npy if reverse else convert_file_npy_to_pcd
        input_ext = "pcd" if reverse else "npy"
        output_ext = "npy" if reverse else "pcd"
        input_files = glob.glob(os.path.join(self.input_directory, f"*.{input_ext}"))
        num_input_files = len(input_files)
        loop_starts = time.time()
        for file_number, input_file in enumerate(input_files):
            now = time.time()
            logger.info(
                f"At iteration {file_number}/{num_input_files}, it has been {now - loop_starts} seconds since the loop started")
            file_base = os.path.splitext(os.path.split(input_file)[-1])[0]
            file_pcd_ext = ".".join([file_base, output_ext])
            output_file = os.path.join(self.output_directory, file_pcd_ext)
            logger.info(f"Input file: {input_file}, Output file: {output_file}.")
            convert_function(input_file, output_file)


def convert_file_npy_to_pcd(input_file, output_file):
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(numpy.load(input_file))
    open3d.io.write_point_cloud(output_file, pcd)


def convert_file_pcd_to_npy(input_file, output_file):
    pcd = open3d.io.read_point_cloud(input_file)
    xyz = numpy.asarray(pcd.points)
    numpy.save(output_file, xyz)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("numpy_dir")
    parser.add_argument("pcd_dir")
    args = parser.parse_args()

    test_numpy_dir = os.path.realpath(os.path.expanduser(args.numpy_dir))
    test_pcd_dir = os.path.realpath(os.path.expanduser(args.pcd_dir))
    dc = DirectoryConverter_npy_to_pcd(test_numpy_dir, test_pcd_dir)
    dc.convert()

    test_created_numpy_dir = os.path.expanduser("~/Data/testData_07_19_19/created_npy")
    dc = DirectoryConverter_npy_to_pcd(test_pcd_dir, test_created_numpy_dir)
    dc.convert(reverse=True)

    for file in os.listdir(test_numpy_dir):
        orig_numpy = numpy.load(os.path.join(test_numpy_dir, file))
        created_numpy = numpy.load(os.path.join(test_created_numpy_dir, file))
        numpy.testing.assert_array_almost_equal(orig_numpy[~numpy.isinf(created_numpy).any(axis=1)], created_numpy[
            ~numpy.isinf(created_numpy).any(axis=1)
        ], decimal=5)

if __name__ == '__main__':
        main()
