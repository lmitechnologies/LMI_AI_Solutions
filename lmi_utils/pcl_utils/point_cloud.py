import logging
import os
import time

import cv2
import numpy as np
# import open3d


import sys
# if '..' not in sys.path:
#     sys.path.append('..')

from scipy.interpolate import griddata
import image_utils.rgb_converter as rbg_converter
from image_utils.img_resize import resize


class PointCloud():
    def __init__(self):
        self.TWO_TO_TWENTYFOURTH_MINUS_ONE = np.power(2, 24)-1
        self.input_path = None
        self.x = None
        self.y = None
        self.z = None
        self.zmin = None
        self.zmax = None
        self.znorm = None
        # x,y_map = pixel coordinates from unique x and y coordinates
        self.x_map = None
        self.y_map = None
        # image x,y values
        self.img_x = None
        self.img_y = None
        # height and width
        self.height = None
        self.width = None

        # Different variants of image
        self.img = None
        self.img_fp = None

    def __normalize_img(self,zmin,zmax):
        # shift and normalize by full scale range of the data
        img_range = zmax-zmin
        img_offset = zmin
        img_norm = (self.img-img_offset)/img_range
        img_norm[img_norm<0]=0
        img_norm[img_norm>=1]=0.999
        return img_norm

    def resample_cloud(self, dx, dy):

        # get min and max
        x_min = self.x.min()
        x_max = self.x.max()
        y_min = self.y.min()
        y_max = self.y.max()

        # set new grid
        x_resamp = np.arange(x_min, x_max, dx)
        y_resamp = np.arange(y_min, y_max, dy)
        grid_y, grid_x = np.meshgrid(y_resamp, x_resamp)

        # interpolate
        xinput = self.img_x.ravel()
        yinput = self.img_y.ravel()
        zinput = self.img_fp.ravel()
        grid_z = griddata(np.vstack((yinput, xinput)).transpose(
        ), zinput, (grid_y, grid_x), method='nearest', fill_value=0.0)

        # reset with resampled image
        self.x = grid_x.transpose().ravel()
        self.y = grid_y.transpose().ravel()
        self.z = grid_z.transpose().ravel()
        self.__make_maps()

    def make_maps_ext(self):
        self.__make_maps()

    def __make_maps(self):
        # find unique x and y values
        x_val, y_val = np.sort(np.unique(self.x)), np.sort(np.unique(self.y))
        W = len(x_val)
        H = len(y_val)
        npts = self.z.shape[0]
        # vector of indices
        x_pos, y_pos = np.arange(len(x_val)), np.arange(len(y_val))
        # dictionary maps value to grid index
        self.x_map, self.y_map = dict(
            zip(x_val, x_pos)), dict(zip(y_val, y_pos))
        # initialize image arrays as grid of zeros
        if W*H == npts:
            self.img_fp = self.z.reshape((H, W))
            self.img_x = self.x.reshape((H, W))
            self.img_y = self.y.reshape((H, W))
        else:
            self.img_fp = np.zeros([H, W], dtype='f')
            self.img_fp[:]=np.nan
            self.img_y, self.img_x = np.meshgrid(y_val, x_val)
            # scan through all points placing
            for index in range(len(self.x)):
                # get the row col for each point
                x_i = self.x_map.get(self.x[index])
                y_i = self.y_map.get(self.y[index])
                z_fp = self.z[index]
                # assign z to floating point img
                self.img_fp[y_i, x_i] = z_fp

        self.img = self.img_fp
        self.height = H
        self.width = W

    def __enhance_contrast(self):
        print('[INFO] Applying local contrast enhancement.')
        bgr = self.img
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        lab_planes = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(4, 4))
        lab_planes[0] = clahe.apply(lab_planes[0])
        lab = cv2.merge(lab_planes)
        bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        self.img = bgr

    def getImage(self, format='rgb'):
        if format == 'rgb':
            img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        elif format == 'gray':
            img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        elif format == 'float':
            # floating point images stored in range 0-1
            img = self.img_fp/np.max(self.img_fp)
        else:
            raise Exception("Sorry, unsupported format")
        return img
        
    def get_PCD(self):
        import open3d
        pcd = open3d.geometry.PointCloud()
        np_points = np.empty((self.x.shape[0], 3))
        np_points[:, 0] = self.x
        np_points[:, 1] = self.y
        np_points[:, 2] = self.z
        pcd.points = open3d.utility.Vector3dVector(np_points)
        return pcd

    def reinitialize_fp_image(self):
        self.img = self.img_fp

    def resize(self, dmax=1000, dmin=1000):
        (H, W) = self.img.shape[:2]
        if W > H:
            if H > dmin:
                self.img = resize(self.img, height=dmin)
                (H, W) = self.img.shape[:2]
            if W > dmax:
                self.img = resize(self.img, width=dmax)
        if H > W:
            if W > dmin:
                self.img = resize(self.img, width=dmin)
                (H, W) = self.img.shape[:2]
            if H > dmax:
                self.img = resize(self.img, height=dmax)

    def pad(self, height, width):
        BLACK = (0, 0, 0)

        (H, W) = self.img.shape[:2]
        if height > H:
            delta=height-H
            pad_bottom = delta//2
            pad_top=delta-pad_bottom
            self.img = cv2.copyMakeBorder(
                self.img, pad_top, pad_bottom, 0, 0, cv2.BORDER_CONSTANT, value=BLACK)
        elif height < H:
            raise Exception(
                f'Converted image height: {H} is greater than output height: {height}.  Increase output height.')

        if width > W:
            delta=width-W
            pad_right = delta//2
            pad_left=delta-pad_right
            self.img = cv2.copyMakeBorder(
                self.img, 0, 0, pad_left, pad_right, cv2.BORDER_CONSTANT, value=BLACK)
        elif width < W:
            raise Exception(
                f'Converted image width: {W} is greater than output width: {width}.  Increase output width.')

    def prune(self, px_xmin=0, px_xmax=1e6, px_ymin=0, px_ymax=1e6, px_zmin=-1e6, px_zmax=1e6):
        ''' (0,0) as top left corner of image
            left-to-right as x direction
            top-to-bottom as y direction
            (x1,y1) as the top-left 
            (x2,y2) as the bottom-right vertex
            roi = im[y1:y2, x1:x2]
        '''
        self.img = self.img[px_ymin:px_ymax, px_xmin:px_xmax]
        self.img[self.img <= px_zmin] = 0
        self.img[self.img >= px_zmax] = 0

    def read_points(self, path, zmin=None, zmax=None, clip_mode=0):
        self.input_path, ext = os.path.splitext(path)
        if ext == '.pcd':
            import open3d
            pcddata = open3d.io.read_point_cloud(path, remove_nan_points=False, remove_infinite_points=False)
            points = np.asarray(pcddata.points)
        elif ext == '.npy':
            points = np.load(path)
        else:
            raise Exception('Input file type not supported.')


        x, y, z = points[:, 0], points[:, 1], points[:, 2]

        # keep all points
        if clip_mode == 0:
            self.z = z
            self.y = y
            self.x = x
        # remove the z values if either zmin or zmax are set
        elif clip_mode==1:
            if (zmin is not None) and (zmax is not None):
                self.z = z[np.logical_and((z >= zmin), (z <= zmax))]
                self.y = y[np.logical_and((z >= zmin), (z <= zmax))]
                self.x = x[np.logical_and((z >= zmin), (z <= zmax))]
            elif (zmax is None) and (zmin is not None):
                self.z = z[z >= zmin]
                self.y = y[z >= zmin]
                self.x = x[z >= zmin]
            elif (zmin is None) and (zmax is not None):
                self.z = z[z <= zmax]
                self.y = y[z <= zmax]
                self.x = x[z <= zmax]
            else:
                self.z = z
                self.y = y
                self.x = x
        # keep all x,y coordinates but relpace outliers with NaN
        elif clip_mode==2:
            if (zmin is not None) and (zmax is not None):
                ind = [np.logical_or((z <= zmin), (z >= zmax))]
            elif (zmax is None) and (zmin is not None):
                ind = [z <= zmin]
            elif (zmin is None) and (zmax is not None):
                ind = [z >= zmax]
            else:
                pass
            zkeep=z[np.squeeze(np.invert(ind))]
            self.zmin=zkeep.min()
            self.zmax=zkeep.max()
            z[ind[0]]=np.nan
            self.z = z
            self.y = y
            self.x = x
        else:
            raise Exception('Unknown import option.  Choose 0: all, 1: clip, 2:replace w/ NaN.')

        # set inf values to nan for consistency
        self.z[self.z == -np.inf] = np.nan
        self.z[self.z <= -1.0e-100] =np.nan
        # filter all nan to determine dataset min/max
        try:
            ind=self.z[np.isnan(self.z)]
            if ind.size != 0:
                ind=np.invert(np.isnan(self.z))
                zkeep=self.z[ind]
            else:
                zkeep=self.z
        except:
            zkeep=self.z
        
        self.zmin=zkeep.min()
        self.zmax=zkeep.max()


        #create the height map
        self.__make_maps()

    def convert_points_to_image(self, color_mapping='rainbow', contrast_enhancement=False, zmin_color=None, zmax_color=None, verbose=False):
        
        zmin_color=self.zmin if zmin_color is None else zmin_color
        zmax_color=self.zmax if zmax_color is None else zmax_color
        img_norm=self.__normalize_img(zmin_color,zmax_color)
        img_norm[np.isnan(img_norm)]=0.0
        if verbose: print(f'[INFO] Normalizing data between {zmin_color} and {zmax_color}')
        try:
            if color_mapping == 'rainbow':
                if verbose: print('[INFO] Converting to rainbow color map.')
                #discretize range
                img_int = (img_norm *self.TWO_TO_TWENTYFOURTH_MINUS_ONE).astype(np.int)
                img = rbg_converter.convert_array_to_rainbow(img_int)
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            elif color_mapping == 'rgb':
                if verbose: print('[INFO] Converting to high-res color map.')
                img_int = (img_norm *self.TWO_TO_TWENTYFOURTH_MINUS_ONE).astype(np.int)
                img = rbg_converter.convert_array_to_rgb(img_int)
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            elif color_mapping == 'gray':
                if verbose: print('[INFO] Converting to grayscale')
                img = (img_norm *255).astype(np.uint8)
                img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        except:
            raise Exception('Invalid color mapping.')
        
        self.img = img_bgr

        # enhance contrast
        if contrast_enhancement:
            self.__enhance_contrast()

    def save_img(self, output_path):
        if self.img is None:
            raise ValueError("Image not calculated.")
        cv2.imwrite(output_path, self.img)

    def save_as_npy(self, fname):
        try:
            arr = np.column_stack((self.x, self.y, self.z))
            np.save(fname, arr)
        except:
            print('Could not write the .npy file.')


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--input', required=True, help='path to input cloud')
    args = vars(ap.parse_args())
    # input_cloud_path=input('Input cloud path:  ')
    input_cloud_path = args['input']
    pc = PointCloud()
    try:
        pc.read_points(input_cloud_path, zmin=0, zmax=40, clip_mode=1)
    except:
        print('Bad path.')
        sys.exit(1)
    tstart = time.time()
    pc.convert_points_to_image(
        colore_mapping='rainbow', contrast_enhancement=True)
    tstop = time.time()
    print(f'[INFO] Time to gen rainbow: {tstop-tstart}s')
    image_rb = pc.img
    pc.reinitialize_fp_image()
    tstart = time.time()
    pc.convert_points_to_image(
        color_mapping='rgb', contrast_enhancement=True)
    tstop = time.time()
    print(f'[INFO] Time to gen rgb: {tstop-tstart}s')
    image_rgb = pc.img
    pc.reinitialize_fp_image()
    tstart = time.time()
    pc.convert_points_to_image(
        color_mapping='gray',contrast_enhancement=True)
    tstop = time.time()
    print(f'[INFO] Time to gen gray: {tstop-tstart}s')
    image_gray = pc.img

    print(
        f'[INFO] Shape Width: {image_gray.shape[1]}, Height: {image_gray.shape[0]}')

    cv2.imshow('Rainbow', image_rb)
    cv2.imshow('RGB', image_rgb)
    cv2.imshow('gray', image_gray)

    cv2.waitKey(0)

    cv2.imwrite('test_1.png', image_rgb)


if __name__ == "__main__":
    main()
