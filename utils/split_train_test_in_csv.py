import csv
import collections
from numpy.core.numeric import full
from sklearn.model_selection import train_test_split
import os

#3rd party library
import csv_utils


def copy_images(data, out_path) -> None:
    import shutil
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    for im_name in data:
        for mask in data[im_name]:
            shutil.copy2(mask.fullpath, out_path)

def split_data_in_csv(path_csv, path_img, test_ratio=0.25, rs=777) -> list:
    data,class_map = csv_utils.load_csv(path_csv, path_img)
    img_names = list(data.keys())
    train_names,test_names = train_test_split(img_names, test_size=test_ratio, random_state=rs)
    #deal with train
    train_data = collections.defaultdict(list)
    for name in train_names:
        train_data[name] = data[name]
    #deal with test
    test_data = collections.defaultdict(list)
    for name in test_names:
        test_data[name] = data[name]
    return train_data,test_data

def generate_split_datasets(train_data, test_data, path_out):
    path_out_train = os.path.join(path_out, 'train')
    copy_images(train_data, path_out_train)
    csv_utils.write_to_csv(train_data, os.path.join(path_out_train, 'labels_train.csv'))

    path_out_test = os.path.join(path_out, 'test')
    copy_images(test_data, path_out_test)
    csv_utils.write_to_csv(test_data, os.path.join(path_out_test, 'labels_test.csv'))


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--path_img', required=True, type=str, help='the path to the image folder')
    ap.add_argument('--path_csv', required=True, type=str, help='the path to the csv file')
    ap.add_argument('--path_out', required=True, type=str, help='the output path to the train dataset')
    ap.add_argument('--test_ratio', default=0.25, help='the ratio of the test dataset')
    args = vars(ap.parse_args())

    train_data, test_data = split_data_in_csv(args['path_csv'], args['path_img'], test_ratio=float(args['test_ratio']))
    generate_split_datasets(train_data, test_data, path_out=args['path_out'])