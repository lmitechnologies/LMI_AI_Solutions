import os, sys
import logging 
from collections import OrderedDict, namedtuple
import tensorrt as trt
import torch
import numpy as np
import warnings
import cv2
import glob
import shutil
import time
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('AnomalyModel')

PASS = 'PASS'
FAIL = 'FAIL'

Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))

class AnomalyModel:
    
    def __init__(self, engine_path):
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            logger.warning('GPU device unavailable. Use CPU instead.')
            self.device = torch.device('cpu')
        with open(engine_path, "rb") as f, trt.Runtime(trt.Logger(trt.Logger.WARNING)) as runtime:
            model = runtime.deserialize_cuda_engine(f.read())
        self.context = model.create_execution_context()
        self.bindings = OrderedDict()
        self.output_names = []
        self.fp16 = False
        for i in range(model.num_bindings):
            name = model.get_tensor_name(i)
            dtype = trt.nptype(model.get_tensor_dtype(name))
            shape = tuple(self.context.get_tensor_shape(name))
            logger.info(f'binding {name} ({dtype}) with shape {shape}')
            if model.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                if dtype == np.float16:
                    self.fp16 = True
            else:
                self.output_names.append(name)
            im = self.from_numpy(np.empty(shape, dtype=dtype)).to(self.device)
            self.bindings[name] = Binding(name, dtype, shape, im, int(im.data_ptr()))
        self.binding_addrs = OrderedDict((n, d.ptr) for n, d in self.bindings.items())

    def preprocess(self, image):
        img = cv2.resize(AnomalyModel.normalize(image), self.bindings['input'].shape[-2:], interpolation=cv2.INTER_AREA)
        input_dtype = np.float16 if self.fp16 else np.float32
        input_batch = np.array(np.repeat(np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0), 1, axis=0), dtype=input_dtype)
        return self.from_numpy(input_batch)

    def warmup(self):
        logger.info("warmup started")
        t0 = time.time()
        shape = [i*2 for i in self.bindings['input'].shape[-2:]]+[3,]
        self.predict(np.zeros(shape))
        logger.info(f"warmup ended - {time.time() - t0}")

    def predict(self, image):
        input_batch = self.preprocess(image)
        self.binding_addrs['input'] = int(input_batch.data_ptr())
        self.context.execute_v2(list(self.binding_addrs.values()))
        outputs = {x:self.bindings[x].data.cpu().numpy() for x in self.output_names}
        return outputs['output']

    @staticmethod
    def convert_trt(onnx_path, out_engine_path, fp16=True, workspace=4096):
        """
        convert an onnx to trt engine
        """
        logger.info(f"converting {onnx_path}...")
        assert(out_engine_path.endswith(".engine")), f"trt engine file must end with '.engine'"
        target_dir = os.path.dirname(out_engine_path)
        if not os.path.isdir(target_dir):
            os.makedirs(target_dir)
        convert_cmd = (f'trtexec --onnx={onnx_path} --saveEngine={out_engine_path}'
                       ' --explicitBatch --inputIOFormats=fp16:chw --outputIOFormats=fp16:chw'
                       f' --workspace={workspace}') + (' --fp16' if fp16 else ' ')
        os.system(convert_cmd)
        os.system(f"cp {os.path.dirname(onnx_path)}/metadata.json {os.path.dirname(out_engine_path)}")

    def from_numpy(self, x):
        return torch.from_numpy(x).to(self.device) if isinstance(x, np.ndarray) else x
    
    @staticmethod
    def normalize(image: np.ndarray) -> np.ndarray:
        """
        Normalize the image to the given mean and standard deviation for consistency with pytorch backbone
        """
        image = image.astype(np.float32)
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        image /= 255.0
        image -= mean
        image /= std
        return image
    
    @staticmethod
    def normalize_anomaly_map(data, mi=None, ma=None):
        """  normalize to [0,1) """
        data_min=mi or data.min()
        data_max=ma or data.max()
        return (data - data_min) / (data_max - data_min + 1e-16)

    @staticmethod
    def annotate(img_original, heat_map_rsz):
        residual_gray = (AnomalyModel.normalize_anomaly_map(heat_map_rsz)*255).astype(np.uint8)
        residual_bgr = cv2.applyColorMap(np.expand_dims(residual_gray,-1), cv2.COLORMAP_TURBO)
        residual_rgb = cv2.cvtColor(residual_bgr, cv2.COLOR_BGR2RGB)
        annot = cv2.addWeighted(img_original, 0.5, residual_rgb, 0.5, 0)
        ind = heat_map_rsz==0
        annot[ind] = img_original[ind]
        return annot

def test(engine_path, images_path, annot_dir,err_thresh=None,annotate_inputs=False):
    """test trt engine"""

    from ad_utils import plot_fig
    from pathlib import Path
    import time
    from scipy.stats import gamma
    import matplotlib.pyplot as plt
    from tabulate import tabulate
    import csv

    # images = glob.glob(f"{images_path}/*.png")
    directory_path=Path(images_path)
    images=list(directory_path.rglob('*.png'))
    logger.info(f"{len(images)} images from {images_path}")
    if not images:
        return
    
    logger.info(f"Loading engine: {engine_path}.")

    pc = AnomalyModel(engine_path)

    out_path = annot_dir
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    pc.warmup()

    proctime = []
 
    img_all,anom_all,fname_all,path_all=[],[],[],[]
    for image_path in images:
        image_path=str(image_path)
        img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        t0 = time.time()
        anom_map = pc.predict(img).astype(np.float32)
        proctime.append(time.time() - t0)
        fname=os.path.split(image_path)[1]
        img_preproc=cv2.resize(img, pc.bindings['input'].shape[-2:], interpolation=cv2.INTER_AREA)
        img_all.append(img_preproc)
        anom_all.append(anom_map)
        fname_all.append(fname)
        path_all.append(image_path)
    
    # Compute & Validate pdf
    anom_sq=np.squeeze(np.array(anom_all))
    data=np.ravel(anom_sq)
    alpha_hat, loc_hat, beta_hat = gamma.fit(data, floc=0)
    x = np.linspace(min(data), max(data), 1000)
    pdf_fitted = gamma.pdf(x, alpha_hat, loc=loc_hat, scale=beta_hat)
    plt.hist(data, bins=100, density=True, alpha=0.7, label='Observed Data')
    plt.plot(x, pdf_fitted, 'r-', label=f'Fitted Gamma')
    plt.legend()
    plt.savefig(os.path.join(annot_dir,'gamma_pdf_fit.png'))
    # Compute and validate cdf
    data_sorted = np.sort(data)
    yvals = np.arange(1, len(data)+1) / len(data)
    # Calculate the theoretical CDF using the fitted gamma distribution
    cdf_theoretical = gamma.cdf(data_sorted, alpha_hat, loc=loc_hat, scale=beta_hat)
    # Plot the ECDF and theoretical CDF
    plt.clf()
    plt.plot(data_sorted, yvals, label='ECDF', marker='.', linestyle='none')
    plt.plot(data_sorted, cdf_theoretical, label='Gamma CDF', color='r')
    plt.legend(loc='upper left')
    plt.xlabel('Value')
    plt.ylabel('Cumulative Probability')
    plt.title('ECDF vs. Fitted Gamma CDF')
    plt.savefig(os.path.join(annot_dir,'gamma_cdf_fit.png'))
    # Compute possible thresholds   
    threshold = np.linspace(min(data), max(data), 10)
    probability_patch = 1 - gamma.cdf(threshold, alpha_hat, loc=loc_hat, scale=beta_hat)
    probability_patch=["{:.{}f}".format(item*100, 4) for item in np.squeeze(probability_patch).tolist()]
    probability_patch=['Prob of Patch Defect']+probability_patch
    probability_sample=['Prob of Sample Defect']
    for t in threshold:
        ind=np.where(anom_sq>t)
        ind_u=np.unique(ind[0])
        percent=len(ind_u)/len(fname_all)*100
        probability_sample.append("{:.{}f}".format(percent, 4))

    threshold=["{:.{}f}".format(item, 4) for item in np.squeeze(threshold).tolist()]
    threshold=['Threshold']+threshold    
    
    tp=[threshold,probability_patch,probability_sample]
    

    tp_print=tabulate(tp, tablefmt='grid')

    logger.info('Threshold options:\n'+tp_print)


    # training_mean=anom_stats.mean()
    # training_std=anom_stats.std()
    # training_H0=[training_mean,training_std]
    # training_max=anom_stats.max()
    if annotate_inputs:
        results=zip(img_all,anom_all,fname_all)
        plot_fig(results,annot_dir)
        
    # get anom stats
    anom_all = np.array(anom_all)
    means = anom_all.mean(axis=0)
    maxs = anom_all.max(axis=0)
    mins = anom_all.min(axis=0)
    
    # write to a csv file
    with open(os.path.join(annot_dir,'stats.csv'), 'w') as csvfile:
        fieldnames = ['fname', 'mean', 'max', 'min']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for data in zip(path_all,means,maxs,mins):
            tmp_dict = {f:d for f,d in zip(fieldnames,data)}
            writer.writerow(tmp_dict)
        
    if proctime:
        proctime = np.asarray(proctime)
        logger.info(f'Min Proc Time: {proctime.min()}')
        logger.info(f'Max Proc Time: {proctime.max()}')
        logger.info(f'Avg Proc Time: {proctime.mean()}')
        logger.info(f'Median Proc Time: {np.median(proctime)}')
    logger.info(f"Test results saved to {out_path}")

def convert(onnx_file, engine_file, fp16=True):
    # engine_out_path = f'{engine_dir}/model.engine'
    AnomalyModel.convert_trt(onnx_file, engine_file, fp16)

if __name__ == '__main__':
    import argparse
    import os

    ap = argparse.ArgumentParser()
    ap.add_argument('-a','--action', default="test", help='Action: convert, test')
    ap.add_argument('-x','--onnx_file', default="/app/onnx/model.onnx", help='Onnx file path.')
    ap.add_argument('-e','--engine_file', default="/app/onnx/engine/model.engine", help='Engine file path.')
    ap.add_argument('-d','--data_dir', default="/app/data", help='Data file directory.')
    ap.add_argument('-o','--annot_dir', default="/app/annotation_results", help='Annot file directory.')
    ap.add_argument('-p','--plot',action='store_true', help='plot the annotated images')

    args = vars(ap.parse_args())
    action=args['action']
    onnx_file=args['onnx_file']
    engine_file=args['engine_file']
    if action=='convert':
        if not os.path.isfile(onnx_file):
            raise Exception('Need a valid onnx file to generate engine.')
        engine_dir,engine_fname=os.path.split(engine_file)
        if not os.path.exists(engine_dir):
            os.makedirs(engine_dir)
        convert(onnx_file,engine_file,fp16=True)

    if action=='test':
        if not os.path.isfile(engine_file):
            raise Exception(f'Error finding {engine_file}. Need a valid engine file to test model.')
        if not os.path.exists(args['annot_dir']):
            os.makedirs(args['annot_dir'])
        test(engine_file, args['data_dir'], args['annot_dir'],err_thresh=None,annotate_inputs=args['plot'])