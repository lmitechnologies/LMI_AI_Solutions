import argparse
import logging
from pcl_utils.point_cloud import PointCloud
import os
import glob

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if __name__ == "__main__":
    # options
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--input', required=True)
    ap.add_argument('-o', '--output', required=True)
    ap.add_argument('--zmax', default=None)
    args = ap.parse_args()

    input_path=args.input
    output_path=args.output
    zmax=args.zmax
    normalize=True
    if zmax is not None:
        zmax=float(args.zmax)
        normalize=False

    pcd = PointCloud()
    if not os.path.isdir(input_path):
            pcd.read_points(input_path)
            pcd.convert_points_to_color_image(normalize=True)
            pcd.save_img(output_path)
    else:
        print(f'[INFO] converting directory of pcds.')
        files=glob.glob(os.path.join(input_path,'*.pcd'))
        for current_file in files:
            print(f'[INFO] Reading: {current_file}')
            fname=os.path.split(current_file)[1]
            fname=os.path.splitext(fname)[0]+'.png'
            fout=os.path.join(output_path,fname)
            print(f'[INFO] writing: {fout}')
            pcd.read_points(current_file)
            pcd.convert_points_to_color_image(normalize=normalize,zmax=zmax)
            pcd.save_img(fout)