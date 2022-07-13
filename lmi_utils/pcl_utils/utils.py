import tensorflow as tf
from pcl_utils.point_cloud import PointCloud

def extract_image(acq_path, W, H):
    '''
    Note: the point_cloud class is incomplete. Some of the functionality is implemented here instead
    This method extracts an image (convert npy -> pcd -> npy image and preprocess) for the loin_sorter model
    '''
    pcd_obj = PointCloud()
    pcd_obj.read_points(acq_path)
    pcd_obj.convert_points_to_color_image()
    image = pcd_obj.getImage('gray')
    image = tf.expand_dims(image, axis=-1)
    image = tf.image.resize_with_pad(image, target_height=H, target_width=W)
    image = image[...,0].numpy() / 255
    return image