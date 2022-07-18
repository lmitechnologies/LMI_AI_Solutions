import glob
import logging
import os
import argparse

from pcl_utils.point_cloud import PointCloud

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class DirectoryConverter(object):
    def __init__(self, input_directory, output_directory):
        self.input_directory = input_directory
        self.output_directory = output_directory

    def convert(
        self, convert_to_color=False, normalize=False, equalize_histograms=False, ext='.png'
    ):
        input_files = glob.glob(os.path.join(self.input_directory, "*.pcd"))
        num_input_files = len(input_files)
        for i, input_file in enumerate(input_files):
            logger.info(f"Processing {input_file}.")
            pcd = PointCloud()
            pcd.read_points(input_file)
            filename, _ = os.path.splitext(os.path.basename(input_file))
            outfile = os.path.join(self.output_directory, filename + ext)
            if ext=='.png':
                if convert_to_color:
                    pcd.convert_points_to_image(color_mapping='rainbow')
                else:
                    pcd.convert_points_to_image(color_mapping='gray') 
                pcd.save_img(outfile)
            elif ext == '.npy':
                pcd.save_as_npy(outfile)
            else:
                raise Exception(f'Unrecognized extension: {ext}. Choose .png or .npy.')
                
            logger.info(f"Converted {i+1}/{num_input_files}")

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--input_path', '-i', required=True)
    ap.add_argument('--output_path', '-o', required=True)
    ap.add_argument('--normalize', type=bool, default=False)
    ap.add_argument('--extension', default='.png')
    args = vars(ap.parse_args())
    inp  = args['input_path']
    out  = args['output_path']
    norm = args['normalize']
    ext  = args['extension']
    assert os.path.exists(inp), f'Path {inp} does not exist'
    if not os.path.exists(out):
        print('[INFO] Creating new directory:', out)
        os.mkdir(out)
    dc = DirectoryConverter(inp, out)
    dc.convert(convert_to_color=False, normalize=norm, ext=ext)

