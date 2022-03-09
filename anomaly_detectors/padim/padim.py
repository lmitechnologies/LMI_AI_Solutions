# 1. Built-in modules
import os
import argparse
import random
import cv2
import logging
import time
import glob

# 2. Third-party modules
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import matplotlib
from matplotlib import pyplot as plt
from skimage import morphology

# 3. Own modules
from padim.data_loader import DataLoader

logging.basicConfig(level=logging.DEBUG)

def plot_histogram(xvec):
    '''
    DESCRIPTION: 
        Helper function for understanding and debugging.  
        Plots histogram of xvec so that we can visualize the embedding vector distribution from each patch in the image 

    ARGS:
        xvec: single dimension numpy array for the error distance across all pixels in the input image
    '''
    # An "interface" to matplotlib.axes.Axes.hist() method
    figure=plt.figure()
    n, bins, patches = plt.hist(x=xvec, bins='auto', color='#0504aa',alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Error Histogram')
    plt.text(23, 45, r'$\mu=15, b=3$')
    maxfreq = n.max()
    # Set a clean upper y-axis limit.
    plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    # plt.show()
    figure.savefig('./test_hist.png')


def plot_fig(predict_results, err_mean, err_std, save_dir, err_ceil_z=1):
    '''
    DESCRIPTION: generate matplotlib figures for inspection results

    ARGS: 
        predict_results: zip object
            image_array: numpy array (batch,dim,dim,3) for all images in dataset
            error_dist_array: numpy array (batch,dim,dim,3) for normalized distances/errors
            fname_array: numpy array (batch) for desriptive filenames for each img/score
        err_mean: mean training error (~0)
        err_std: std training error
        save_dir: path to save directory
        err_ceil_z: z score for heat map normalization
    '''

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # Assume normalized error distance
    ERR_FLOOR = 0

    for img,err_dist,fname in predict_results:
        fname=fname.decode('ascii')
        fname=os.path.splitext(fname)[0]
        err_dist=np.squeeze(err_dist)
        err_min=err_dist.min()
        err_max=err_dist.max()
        mask=(err_dist-err_min)/(err_max-err_min)
        kernel = morphology.disk(4)
        mask = morphology.opening(mask, kernel)
        mask *= 255
        # compute z score of error data using training statistics
        heat_map=(err_dist-err_mean)/err_std
        # Disregard errors below an error floor
        heat_map[heat_map<ERR_FLOOR]=0
        # Map errors greater than error ceiling to the error ceiling
        heat_map[heat_map>err_ceil_z]=err_ceil_z
        # Normalize z scores to the error ceiling for plotting
        heat_map=heat_map/err_ceil_z
        heat_map=np.nan_to_num(heat_map)
        fig_img, ax_img = plt.subplots(1, 3, figsize=(12, 3))
        fig_img.subplots_adjust(right=0.9)
        for ax_i in ax_img:
            ax_i.axes.xaxis.set_visible(False)
            ax_i.axes.yaxis.set_visible(False)
        ax_img[0].imshow(img.astype(int))
        ax_img[0].title.set_text('Image')
        # ax = ax_img[1].imshow(heat_map, cmap='jet')
        ax_img[1].imshow(img.astype(int), cmap='gray', interpolation='none')
        ax=ax_img[1].imshow(heat_map, cmap='jet', alpha=0.5, interpolation='none',vmin=0,vmax=1)
        ax_img[1].title.set_text('Predicted Heat Map')
        ax_img[2].imshow(mask.astype(int), cmap='gray')
        ax_img[2].title.set_text('Predicted Mask')
        left = 0.92
        bottom = 0.15
        width = 0.015
        height = 1 - 2 * bottom
        rect = [left, bottom, width, height]
        cbar_ax = fig_img.add_axes(rect)
        cb = plt.colorbar(ax, shrink=0.6, cax=cbar_ax, fraction=0.046)
        cb.ax.tick_params(labelsize=8)
        font = {
            'family': 'serif',
            'color': 'black',
            'weight': 'normal',
            'size': 8,
        }
        cb.set_label('Anomaly Score', fontdict=font)
        filepath=os.path.join(save_dir,f'{fname}_annot.png')
        folder=os.path.split(filepath)[0]
        if not os.path.exists(folder):
            os.makedirs(folder)
        fig_img.savefig(filepath, dpi=100)
        plt.close()


class PaDiM(object):

    def __init__(self,GPU_memory=4096):
        self.net=None
        self.mean=None
        self.cov_inv=None
        self.c=None
        self.img_shape=None
        self.random_vector_indices=None
        self.training_mean_dist=None
        self.training_std_dist=None
        self.training_min_dist=None
        self.training_max_dist=None
        logging.info(f'Setting GPU memory limit to {GPU_memory} MB')
        self.set_gpu_memory(GPU_memory)
        

    def set_gpu_memory(self,mem_limit):
        '''
        DESCRIPTION: Configure GPU for training or inference.

        ARGS: GPU memory limit in MB
        '''
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
            try:
                tf.config.set_logical_device_configuration(
                    gpus[0],
                    [tf.config.LogicalDeviceConfiguration(memory_limit=mem_limit)])
                logical_gpus = tf.config.list_logical_devices('GPU')
                logging.info(f'{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs')
            except RuntimeError as e:
                # Virtual devices must be set before GPUs have been initialized
                print(e)

    def embedding_net(self, net_type, shape=None, layer_defs={}):
        '''
        DESCRIPTION: Selects and configures ImageNet pretrained backbone network.

        ARGS: 
            net_type: 'res' -> resnet50, 'eff' -> efficientnetb7

        MODIFIES:
            self.net with selected/configured network

        RETURNS:
            shape_embed -> embedding vector shape H,W
            shape_img -> training image shape

        '''
        if net_type == 'res':
            name_l1=layer_defs.pop('layer1','pool1_pool')
            name_l2=layer_defs.pop('layer2','conv2_block1_preact_relu')
            name_l3=layer_defs.pop('layer3','conv3_block1_preact_relu')

            # resnet 50v2
            if (shape != (224,224) ) and (shape is not None):
                raise Exception('Resnet models require 224x224 image shape.')
            h, w, ch = 224, 224, 3 
            input_tensor = tf.keras.layers.Input([224, 224, 3], dtype=tf.float32)
            x = tf.keras.applications.resnet_v2.preprocess_input(input_tensor)
            model = tf.keras.applications.ResNet50V2(include_top=False, weights='imagenet', input_tensor=x, pooling=None)

            layer1 = model.get_layer(name=name_l1).output
            layer2 = model.get_layer(name=name_l2).output
            layer3 = model.get_layer(name=name_l3).output

        elif net_type == 'eff':
            name_l1=layer_defs.pop('layer1','stem_activation')
            name_l2=layer_defs.pop('layer2','block2a_activation')
            name_l3=layer_defs.pop('layer3','block4a_activation')


            # efficient net B7\
            if shape is None:
                h, w, ch = 224, 224, 3
            else:
                h=shape[0]
                w=shape[1] 
                ch=3

            input_tensor = tf.keras.layers.Input([h, w, ch], dtype=tf.float32)
            x = tf.keras.applications.efficientnet.preprocess_input(input_tensor)
            model = tf.keras.applications.EfficientNetB7(include_top=False, weights='imagenet', input_tensor=x,
                                                        pooling=None)

            #layer1 = model.get_layer(name='stem_activation').output
            layer1 = model.get_layer(name=name_l1).output
            layer2 = model.get_layer(name=name_l2).output
            layer3 = model.get_layer(name=name_l3).output

        else:
            raise Exception("[NotAllowedNetType] network type is not allowed ")

        model.trainable = False
        with open ('model_summary.txt','w') as fh:
        # Pass the file handle in as a lambda function to make it callable
            model.summary(print_fn=lambda x: fh.write(x + '\n'))
        
        shape_embed = (layer1.shape[1], layer1.shape[2], layer1.shape[3] + layer2.shape[3] + layer3.shape[3])
        shape_img=(h,w,ch)

        self.net=tf.keras.Model(model.input, outputs=[layer1, layer2, layer3])

        return shape_embed, shape_img

    def upsample_concatenate_output_layers(self,output_layers):
        '''
        DESCRIPTION: Upsamples and concatenates output layers.  First layer sets resolution.  Subsequent 
        layers use nearest neighbor interpolation to set constant value across each upsampled patch.

        ARGS:
            output_layers -> list of tf.tensors for network output layers, layer0=highest resolution

        RETURNS:
            embedding_vectors -> tf.tensor for upsampled and contatenated embedding vectors
        
        '''
        # set resolution based on first layer
        embedding_vectors=output_layers[0]
        _,h0,w0,_=embedding_vectors.shape
        # upsample and concatenate all subsequent layers
        if len(output_layers)>1:
            for layer_i in output_layers[1:]:
                _,hi,wi,_=layer_i.shape
                s = int(h0 / hi)
                # upsample using nearest interpolation to replicate values in each grid cell
                up_sample=tf.keras.layers.UpSampling2D(size=(s,s),interpolation='nearest')(layer_i)
                embedding_vectors=tf.concat([embedding_vectors,up_sample],axis=-1)
        return embedding_vectors

    def efficient_mahalanobis(self,embedding_vectors: tf.Tensor, mean: tf.Tensor, cov_inv: tf.Tensor, shape : tuple) -> tf.Tensor:
        '''
        DESCRIPTION: 
            Mahalanobis distance calculator
            https://github.com/scipy/scipy/blob/703a4eb497900bdb805ca9552856672c7ef11d21/scipy/spatial/distance.py#L285
        ARGS:
            embedding_vectors (np.ndarray)
            mean (np.ndarray): patchwise mean for reference feature embeddings
            cov_inv (np.ndarray): patchwise inverse of covariance matrix for reference feature embeddings
            shape (tuple): input shape of the feature embeddings
        RETURNS:
            tf.Tensor: distance from the reference distribution
        '''
        B, H, W, C = shape
        delta = tf.expand_dims(embedding_vectors - mean, -1)
        res = tf.matmul(delta, tf.matmul(cov_inv, delta), transpose_a=True)
        dist_list = tf.squeeze(tf.sqrt(res))
        dist_list = tf.reshape(dist_list, shape=(B, H, W))

        return dist_list

    def export_tensors(self,fname="padim.tfrecords"):
        '''
        DESCRIPTION: Export key tf.tensors: mean, cov_inv, random_vector_indices, number of patches per image, embedding vector depth

        ARGS:
            fname -> exported .tfrecords file 
        '''

        def _bytes_feature(value):
            if isinstance(value,type(tf.constant(0))):
                # print('[INFO] Converting value to numpy.')
                value=value.numpy()
            if type(value).__name__=='ndarray':
                pass
            else:
                if not isinstance(value,list):
                    value=[value]
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

        def _float_feature(value):
            if type(value).__name__=='ndarray':
                pass
            else:
                if not isinstance(value,list):
                    value=[value]
            return tf.train.Feature(float_list=tf.train.FloatList(value=value))

        def _int64_feature(value):
            if type(value).__name__=='ndarray':
                pass
            else:
                if not isinstance(value,list):
                    value=[value]
            return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

        
        img_h,img_w=self.img_shape
        ncells,ev_h,ev_w=self.cov_inv.shape
        assert(ev_w==ev_h)
        cov_inv_preproc=self.cov_inv.numpy().reshape(ncells*ev_w*ev_h)
        mean_preproc=self.mean.numpy().reshape(ncells*ev_w)
        
        random_ind_preproc=self.random_vector_indices.numpy()
        training_mean_dist=self.training_mean_dist
        training_std_dist=self.training_std_dist

        logging.debug(f'Convert image h to feature.')
        intfeature_img_h=_int64_feature(img_h)
        logging.debug(f'Convert image w to feature.')
        intfeature_img_w=_int64_feature(img_w)
        logging.debug(f'Convert ncells to feature.')
        intfeature_ncells=_int64_feature(ncells)
        logging.debug(f'Convert c to feature.')
        intfeature_c=_int64_feature(ev_w)
        logging.debug(f'Convert inverse covariance matrix to feature.')
        floatfeature_cov_inv=_float_feature(cov_inv_preproc)
        logging.debug(f'Convert mean to feature.')
        floatfeature_mean=_float_feature(mean_preproc)
        logging.debug(f'Convert random_ind to feature.')
        intfeature_randind=_int64_feature(random_ind_preproc)
        logging.debug(f'Convert training mean distance to feature.')
        floatfeature_tr_err_mean=_float_feature(training_mean_dist)
        logging.debug(f'Convert training std distance to feature.')
        floatfeature_tr_err_std=_float_feature(training_std_dist)

        feature={
            "img_h":intfeature_img_h,
            "img_w":intfeature_img_w,
            "ncells":intfeature_ncells,
            "c":intfeature_c,
            "cov_inv":floatfeature_cov_inv,
            "mean":floatfeature_mean,
            "rvd":intfeature_randind,
            "tr_mean_dist":floatfeature_tr_err_mean,
            "tr_std_dist":floatfeature_tr_err_std,
        }

        logging.debug(f'Combining features.')
        features=tf.train.Features(feature=feature)
        proto=tf.train.Example(features=features)
        logging.debug(f'Serializing features.')
        record_bytes=proto.SerializeToString()
        logging.debug(f'Writing tfrecord.')
        with tf.io.TFRecordWriter(fname) as file_writer:
            file_writer.write(record_bytes)


    def import_tfrecords(self,tfrecord_file):
        '''
        DESCRIPTION: import key tf.tensors: mean, cov_inv, random_vector_indices, number of patches per image, embedding vector depth.  Reshapes tensors for distance calculations.

        MODIFIES:
            self.mean -> tf.tensor (HxW,Ch) for vector mean across image samples.  Each row maps to an image patch.  Each column maps to a channel from the embedding vector.
            self.cov_inv -> tf.tensor (HxW,Ch,Ch) for vector covariance across image samples. Each batch maps to an image patch.  Each row/col maps to embedding vector covariance.
            self.random_vector_indices -> tf.tensor (1xc) vector identifying which embedding vector components to keep
        '''

        def decode_fn(record_bytes):
            schema={
                "img_h":tf.io.FixedLenFeature([], dtype=tf.int64),
                "img_w":tf.io.FixedLenFeature([], dtype=tf.int64),
                "ncells":tf.io.FixedLenFeature([], dtype=tf.int64),
                "c":tf.io.FixedLenFeature([], dtype=tf.int64),
                "cov_inv":tf.io.FixedLenSequenceFeature([], dtype=tf.float32,allow_missing=True,default_value=0),
                "mean":tf.io.FixedLenSequenceFeature([], dtype=tf.float32,allow_missing=True,default_value=0),
                "rvd":tf.io.FixedLenSequenceFeature([], dtype=tf.int64,allow_missing=True,default_value=0),
                "tr_mean_dist":tf.io.FixedLenFeature([], dtype=tf.float32),
                "tr_std_dist":tf.io.FixedLenFeature([], dtype=tf.float32),
                }
            parsed=tf.io.parse_single_example(record_bytes,schema)
            return parsed
        
        dataset=tf.data.TFRecordDataset(tfrecord_file)
        parsed_dataset=dataset.map(decode_fn)
        
        logging.debug(f'Parsed data from {tfrecord_file}: {parsed_dataset}')
        for example in parsed_dataset.take(1):
            img_h_loaded=example['img_h'].numpy()
            img_w_loaded=example['img_w'].numpy()
            ncells_loaded=example['ncells'].numpy()
            c_loaded=example['c'].numpy()
            mean_loaded=example['mean']
            cov_inv_loaded=example['cov_inv']
            random_ind_loaded=example['rvd']
            tr_mean_dist_loaded=example['tr_mean_dist'].numpy()
            tr_std_dist_loaded=example['tr_std_dist'].numpy()

        self.img_shape=(img_h_loaded,img_w_loaded)
        self.mean=tf.reshape(mean_loaded,(ncells_loaded,c_loaded))
        self.cov_inv=tf.reshape(cov_inv_loaded,(ncells_loaded,c_loaded,c_loaded))
        self.random_vector_indices=random_ind_loaded
        self.training_mean_dist=tr_mean_dist_loaded
        self.training_std_dist=tr_std_dist_loaded

    def filter_embedding_vector(self,embedding_flat_vectors_in: tf.Tensor) -> tf.Tensor:
        
        # Slice embedding vectors using random vector indices
        # TODO: find a better way to do this.  Transpose moves the channel column to the 0-index so we can use tf.gather() with the index array, then transpose back
        embedding_flat_vectors_out=tf.transpose(embedding_flat_vectors_in,perm=[2,0,1])
        embedding_flat_vectors_out=tf.gather(embedding_flat_vectors_out,indices=self.random_vector_indices)
        embedding_flat_vectors_out=tf.transpose(embedding_flat_vectors_out,perm=[1,2,0])
        return embedding_flat_vectors_out
        
    def padim_train(self,trainingdata_obj, c=None, net_type='res', is_plot=True, err_ceil_z=None, layer_names={}):
        '''
        DESCRIPTION: Train a new PaDiM model

        ARGS:
            trainingdata_obj: tf.data.dataset that includes full resolution images and filenames
            img_shape: image shape used by PaDiM model
            c: random embedding vector depth
            net_type: options for supported nets.  Currently support: resnet50-> res, efficientnetB7->eff
            is_plot: generate figures for training data
            layer_names: layers used for embedding vectors

        MODIFIES: 
            self.mean: tf.tensor for mean of each vector component across patches and image samples
            self.cov_inv: tf.tensor for component covariance across patches, image samples
            self.random_vector_indices: indices for random components used for comparison with training distribution
        '''
        # Method specific modules:
        import tensorflow_probability as tfp

        # Preprocess training data
        logging.info(f'Preprocessing dataset for training.')
        
        trainingdataset = trainingdata_obj.dataset

        self.img_shape=trainingdata_obj.img_shape
        self.embedding_net(net_type, self.img_shape, layer_names)

        embedding_vectors_train = []
        training_images=[]
        
        logging.info(f'Extracting embedding vectors from training data.')
        for x,fname in trainingdataset:
            fname_str=' '.join([elem.decode('ascii') for elem in fname.numpy()])
            logging.info(f'Generating embedding vector for: {fname_str}')
            output_layers=self.net(x)
            embedding_vectors_train.append(self.upsample_concatenate_output_layers(output_layers))
            training_images.append(x)
        
        embedding_vectors_train=tf.concat(embedding_vectors_train,axis=0)
        training_images=tf.concat(training_images,axis=0)

        B, H, W, C = embedding_vectors_train.shape
        embedding_flat_vectors_train = tf.reshape(embedding_vectors_train, (B, H * W, C))
        random_ind=np.arange(C)
        if c is None:
            self.c=C
        else:
            self.c=c
        
        # Randomize vector indices
        # TODO: replace with PCA
        np.random.shuffle(random_ind)
        random_ind=tf.convert_to_tensor(random_ind)
        random_ind=random_ind[0:self.c]
        self.random_vector_indices=random_ind

        # Filter embedding vector using random vector indices
        embedding_flat_vectors_train_rd=self.filter_embedding_vector(embedding_flat_vectors_train)

        # get the mean and covariance matrix for the reference data
        mean = tf.reduce_mean(embedding_flat_vectors_train_rd, axis=0) # shape (H*W, C)
        I = tf.eye(self.c)

        # Calculate the covariance of feature vectors for each patch
        mult = 1 if B==1 else B/(B-1)
        cov = mult*tfp.stats.covariance(embedding_flat_vectors_train_rd) + 0.01*I  # shape (H*W, C, C)

        # Inverse of covariance matrix
        # Mahalanobis distance calculation needs inverse of covariance matrix
        cov_inv = tf.linalg.inv(cov, adjoint=False, name=None)

        self.mean = mean
        self.cov_inv=cov_inv
        
        # Compute training data statistics for error thresholds
        # run PaDiM predict
        image_tensor,dist_tensor,fname_tensor=self.padim_predict(trainingdataset)
        self.training_mean_dist=tf.math.reduce_mean(dist_tensor).numpy()
        self.training_std_dist=tf.math.reduce_std(dist_tensor).numpy()        
        
        #%% Validate Training Model
        if err_ceil_z is not None:
            # Generate numpy arrays for visualization
            err_dist_array=dist_tensor.numpy()
            image_array=image_tensor.numpy()
            fname_array=fname_tensor.numpy()
            predict_results=zip(image_array,err_dist_array,fname_array)
            # Plot results
            if is_plot:
                plot_fig(predict_results,self.training_mean_dist,self.training_std_dist,'./validation',err_ceil_z=err_ceil_z)


    def padim_predict(self,dataset):
        '''
        DESCRIPTION: Predict PaDiM model

        ARGS:
            img: tf.tensor for input image OR path containing images

        MODIFIES: 
            self.mean: tf.tensor for mean of each vector component across patches and image samples
            self.cov_inv: tf.tensor for component covariance across patches, image samples
            self.random_vector_indices: indices for random components used for comparison with training distribution

        RETURNS:
            image_tensor: image tf.tensor (b,h,w,ch)
            dist_tensor: error distance tf.tensor (b,h,w,ch)
            fname_tensor: filename tf.tensor (b)
        '''
        if isinstance(dataset,type('string')):
            predictdata=DataLoader(path_base=dataset, img_shape=self.img_shape, batch_size=1, shuffle=False)
            dataset=predictdata.dataset
        else:
            pass
            
        proctime=[]
        image_list=[]
        dist_list=[]
        fname_list=[]  
        for x,fname in dataset:
            if type(fname.numpy())==type(np.asarray(0)):
                fname_decode=[elem.numpy().decode('ascii') for elem in fname]
            else:
                fname_decode=fname.numpy().decode('ascii')
            fname_list.append(fname_decode)
            # fname_str=' '.join(fname_decode)
            logging.info(f'Processing images {fname_decode}')
            t0=time.time()
            if len(x.shape)<4:
                x=tf.expand_dims(x,0)
            image_list.append(x)
            output_layers=self.net(x)
            test_layers=self.upsample_concatenate_output_layers(output_layers)
            B, H, W, C = test_layers.shape
            embedding_flat_vectors_test = tf.reshape(test_layers, (B, H * W, C))
            embedding_flat_vectors_test_rd=self.filter_embedding_vector(embedding_flat_vectors_test)
            # Compute the distance
            dist_tensor_x=self.efficient_mahalanobis(embedding_flat_vectors_test_rd, self.mean, self.cov_inv, (B, H, W, self.c))
            # Apply Gaussion Filtering
            dist_tensor_x=tf.expand_dims(dist_tensor_x,-1)
            dist_tensor_x=tf.image.resize(dist_tensor_x,self.img_shape)
            dist_tensor_x=tfa.image.gaussian_filter2d(dist_tensor_x,filter_shape=(3,3))
            # Aggregate tensors in batch
            dist_list.append(dist_tensor_x)
            t1=time.time()
            tdel=(t1-t0)/B
            logging.info(f'Proc Time: {tdel}')
            proctime.append(tdel)
        
        image_tensor=tf.concat(image_list,axis=0)
        dist_tensor=tf.concat(dist_list,axis=0)
        fname_tensor=tf.concat(fname_list,axis=0)

        proctime=np.asarray(proctime)
        logging.info(f'Min Proc Time: {proctime.min()}')
        logging.info(f'Max Proc Time: {proctime.max()}')
        logging.info(f'Avg Proc Time: {proctime.mean()}')

        return image_tensor,dist_tensor,fname_tensor
      

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=42, type=int, help='What seed to use')
    parser.add_argument("--cprime", default=250, type=int, help='Random sampling dimension')
    parser.add_argument("--data_path", default='./data/test_data_transistor/expected', type=str, help="Input Data Path.")
    parser.add_argument("--dim", default=224, type=int, help="Image shape: h=dim, w=dim")
    parser.add_argument("--batch_size", default=40, type=int, help="What batch size to use")
    parser.add_argument("--is_plot", default=True, type=bool, help="Whether to plot or not")
    parser.add_argument("--net", default='res', type=str, help="Which embedding network to use", choices=['eff', 'res'])

    args = parser.parse_args()


    if args.seed > -1:
        np.random.seed(args.seed)
        random.seed(args.seed)
        tf.random.set_seed(args.seed)

    img_shape=(args.dim,args.dim)

    dataloader=DataLoader(path_base='./data/padim/expected', img_shape=img_shape, batch_size=args.batch_size)

    padim=PaDiM(GPU_memory=20000)

    if not os.path.exists('./saved_model'):
        layers={'layer1':'conv3_block1_preact_relu','layer2':'conv4_block1_preact_relu','layer3':'conv5_block1_preact_relu'}
        padim.padim_train(dataloader, c=args.cprime,layer_names=layers,err_ceil_z=10)
        os.makedirs('./saved_model')
        padim.net.save('./saved_model/saved_model')
        padim.export_tensors(fname='./saved_model/padim.tfrecords')
    else:
        padim.import_tfrecords('./saved_model/padim.tfrecords')
        padim.net=tf.keras.models.load_model('./saved_model/saved_model')

    
    image_tensor,dist_tensor,fname_tensor=padim.padim_predict('./data/padim/anomaly')
    err_dist_array=dist_tensor.numpy()
    image_array=image_tensor.numpy()
    fname_array=fname_tensor.numpy()
    prediction_results=zip(image_array,err_dist_array,fname_array)
    plot_fig(prediction_results,padim.training_mean_dist,padim.training_std_dist,'./validation',err_ceil_z=20)