import argparse
import os
import yaml
from padim.padim import PaDiM
from padim.data_loader import DataLoader


def train_padim(path_data:str, config:dict, path_out:str, imsz:tuple, cprime=200, batch_sz=32, gpu_mem=2048):
    
    padim=PaDiM(GPU_memory=gpu_mem)
    dataloader=DataLoader(path_base=path_data,img_shape=imsz,batch_size=batch_sz, img_exts=['png','jpg'])

    layerconfig={'layer1':config['layer1'],'layer2':config['layer2'],'layer3':config['layer3']}
    padim.padim_train(dataloader,c=cprime,net_type=config['name'],layer_names=layerconfig,is_plot=False)      
    if not os.path.isdir(os.path.join(path_out,'saved_model')):
        os.makedirs(os.path.join(path_out,'saved_model'))
    padim.net.save(os.path.join(path_out,'saved_model','saved_model'))
    padim.export_tensors(fname=os.path.join(path_out,'saved_model','padim.tfrecords'))
    print('Done')


if __name__== '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_data', required=True, help='the path to training data')
    parser.add_argument('--path_out', required=True, help='the path to the saved model')
    parser.add_argument('--config_yaml', required=True, help='the yaml file specifies the layers configs')
    parser.add_argument('--imsz', default="224,224", help='comma separated image dimension: w,h. default=224,224')
    parser.add_argument('--n', default=200, type=int, help='the number of vectors to randomly draw, default=200')
    parser.add_argument('--batch_sz', default=8, type=int, help='batch size, default=8')
    parser.add_argument('--gpu_mem', default=16384, type=int, help='gpu memory limit, default=16384')
    args = parser.parse_args()
    
    with open(args.config_yaml, "r") as stream:
        dt = yaml.safe_load(stream)
    imsz = tuple(map(int,args.imsz.split(',')))

    train_padim(args.path_data, dt['backbone'], args.path_out, imsz, args.n, args.batch_sz, args.gpu_mem)
