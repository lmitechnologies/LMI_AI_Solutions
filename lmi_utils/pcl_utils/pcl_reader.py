import numpy as np
import pandas

import open3d


class PCLReader(object):
    def __init__(self):
        self.initial_rows = 11

    # DEPRICATE AND REMOVE THIS METHOD!
    def read_pcl(self, path_to_pcl):
        # This method expects a csv and returns a pandas df: to get the numpy array, take the ".values" of it
        return pandas.read_csv(
            path_to_pcl,
            skiprows=self.initial_rows,
            header=None,
            names=("x", "y", "z"),
            delim_whitespace=True,
        )

    def read_open_3d_format(self, path_to_pcd_binary):
        # This method expects a pcl binary and return a numpy array
        pcddata = open3d.read_point_cloud(path_to_pcd_binary)
        return np.asarray(pcddata.points)
