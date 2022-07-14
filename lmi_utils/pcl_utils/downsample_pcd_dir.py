import numpy as np
import cv2
from tqdm import tqdm
import open3d
from pcl_utils.point_cloud import PointCloud
import argparse
import os

def main(inp, out, rate):
    input_pcds = os.listdir(inp)
    for i in tqdm(range(len(input_pcds))):
        inp_path = os.path.join(inp, input_pcds[i])
        pcd_obj = PointCloud()
        pcd_obj.read_points(inp_path)
        pcd_obj.resample_cloud(2, 2)
        downpcd = pcd_obj.get_PCD()
        out_path = os.path.join(out, input_pcds[i])
        open3d.io.write_point_cloud(out_path, downpcd)

if __name__ == '__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--input_path', required=True)
    ap.add_argument('--output_path', required=True)
    ap.add_argument('--sampling_rate', type=float, required=True)
    args = vars(ap.parse_args())
    inp  = args['input_path']
    out  = args['output_path']
    rate = args['sampling_rate']
    assert os.path.exists(inp), f'Path {inp} does not exist'
    if not os.path.exists(out):
        print('[INFO] Creating new directory:', out)
        os.mkdir(out)
    assert rate > 0, f'Received negative sampling rate of {rate}'
    main(inp, out, rate)