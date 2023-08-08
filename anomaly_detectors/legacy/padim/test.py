import argparse
import os
import yaml
import tensorflow as tf
from padim.padim import PaDiM, plot_fig
from padim.data_loader import DataLoader


def test_padim(testdata_path:str, outpath:str, modelpath:str, err_thresh:float, gpu_mem:int):
    padim=PaDiM(GPU_memory=gpu_mem)
    padim.import_tfrecords(os.path.join(modelpath,'padim.tfrecords'))
    padim.net=tf.keras.models.load_model(os.path.join(modelpath,'saved_model'))
    image_tensor,dist_tensor,fname_tensor=padim.predict(testdata_path)
    err_dist_array=dist_tensor.numpy()
    image_array=image_tensor.numpy()
    fname_array=fname_tensor.numpy()
    prediction_results=zip(image_array,err_dist_array,fname_array)
    if not os.path.isdir(outpath):
        os.makedirs(outpath)
    plot_fig(prediction_results,padim.training_mean_dist,padim.training_std_dist,outpath,err_thresh=err_thresh)


if __name__== '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_data', required=True, help='the path to the testing data')
    parser.add_argument('--path_model', required=True, help='the path to the saved model')
    parser.add_argument('--path_out', required=True, help='the output path')
    parser.add_argument('--thres_err', default=20, type=int, help='the error threshold, default=20')
    parser.add_argument('--gpu_mem', default=4096, type=int, help='gpu memory limit, default=4096')
    args = parser.parse_args()

    test_padim(args.path_data, args.path_out, args.path_model, args.thres_err, args.gpu_mem)
