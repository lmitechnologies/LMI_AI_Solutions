from label_utils import csv_utils
import os
import argparse
import glob

def combine_csv(csv_file_list, output_dir):
    all_shapes = {}
    for csv_file in csv_file_list:
        shapes, _ = csv_utils.load_csv(csv_file)
        all_shapes.update(shapes)
    csv_utils.write_to_csv(shapes=all_shapes, filename=f"{os.path.join(output_dir, 'combined.csv')}")
    return all_shapes

def main():
    parser = argparse.ArgumentParser(description='Combine multiple csv files')
    parser.add_argument('--input-dir', type=str, help='Input directory containing csv files')
    parser.add_argument('--output-dir', type=str, help='Output directory to save the combined csv file')
    args = parser.parse_args()
    csv_files = glob.glob(f"{args.input_dir}/*.csv")
    combine_csv(csv_files, args.output_dir)
    
if __name__ in '__main__':
    main()