from PIL import Image
import pickle
import numpy
from os import listdir, makedirs
from os.path import isfile, join, isdir

class GadgetSurfaceUtils():

    SCHEMA_ID: str = "gadget3d"
    VERSION: int = 1

    def pkl_2_npy(self, source_path, destination_path, intensity=False):
        files = [f for f in listdir(source_path) if isfile(join(source_path, f)) and ".gadget3d.pickle" in f]

        for file in files:
            print(join(source_path, file))

            with open(join(source_path, file), "rb") as f:
                content = pickle.load(f)

            profile = content["profile_array"]
            if profile.dtype == numpy.int16:
                profile = profile.view(numpy.uint16) + numpy.uint16(32768)
            
            numpy.save(join(destination_path, file.replace('.gadget3d.pickle', '.npy')), profile)

            if intensity:
                try:
                    if content["intensity_array"] is not None:
                        numpy.save(join(destination_path, file.replace('.gadget3d.pickle', '-intensity.npy')), content["intensity_array"])
                except KeyError:
                    continue
            
    def pkl_2_png(self, source_path, destination_path, intensity=False):
        files = [f for f in listdir(source_path) if isfile(join(source_path, f)) and ".gadget3d.pickle" in f]

        for file in files:
            print(join(source_path, file))

            with open(join(source_path, file), "rb") as f:
                content = pickle.load(f)

            profile = content["profile_array"]
            if profile.dtype == numpy.int16:
                profile = profile.view(numpy.uint16) + numpy.uint16(32768)
            
            image = Image.fromarray(profile)
            image.save(join(destination_path, file.replace('.gadget3d.pickle', '.png')))
            
            if intensity:
                try:
                    if content["intensity_array"] is not None:
                        image = Image.fromarray(content["intensity_array"])
                        image.save(join(destination_path, file.replace('.gadget3d.pickle', '-intensity.png')))
                except KeyError:
                    continue
                
    def pkl_2_pcd(self, source_path, destination_path):
        import open3d
        files = [f for f in listdir(source_path) if isfile(join(source_path, f)) and ".gadget3d.pickle" in f]

        for file in files:
            print(join(source_path, file))

            with open(join(source_path, file), "rb") as f:
                content = pickle.load(f)

            profile = content["profile_array"]
            resolution = content["metadata"]["resolution"]
            offset = content["metadata"]["offset"]

            shape = profile.shape
            
            # np_z = profile.reshape((shape[0]* shape[1]))
            np_z = numpy.empty(shape[0]*shape[1])
            np_x = numpy.empty(shape[0]*shape[1])
            np_y = numpy.empty(shape[0]*shape[1])
            i = 0
            for y in range(0, shape[1]):
                for x in range(0, shape[0]):
                    np_x[i] = offset[0] + x * resolution[0]
                    np_y[i] = offset[1] + y * resolution[1]
                    np_z[i] = offset[2] + profile[x][y] * resolution[2]
                    i += 1

            np_points = numpy.empty((shape[0]*shape[1], 3))
            np_points[:, 0] = np_x
            np_points[:, 1] = np_y
            np_points[:, 2] = np_z

            pcd = open3d.geometry.PointCloud()
            pcd.points = open3d.utility.Vector3dVector(np_points)
            open3d.io.write_point_cloud(join(destination_path, file.replace(".gadget3d.pickle", ".pcd")), pcd)

    def npy_2_pkl(self, source_path, destination_path):
        files = [f for f in listdir(source_path) if isfile(join(source_path, f)) and ".npy" in f]

        for file in files:
            print(join(source_path, file))

            npy_arr = numpy.load(join(source_path, file))
            
            if npy_arr.dtype == numpy.uint16:
                npy_arr = npy_arr.view(numpy.int16) + numpy.int16(-32768) 
            elif npy_arr.dtype == numpy.int32:
                npy_arr = (npy_arr - 32768).astype(numpy.int16) 
            
            content = { 
                "metadata": {
                    "schema": self.SCHEMA_ID,
                    "version": self.VERSION, 
                    "resolution": 1, 
                    "offset": 0, 
                }, 
                "profile_array": npy_arr,
                "intensity_array": None,
            }

            with open(join(destination_path, file.replace(".png", ".gadget3d.pickle")), "wb") as f:
                pickle.dump(content, f, protocol=4)

    def png_2_pkl(self, source_path, destination_path):
        files = [f for f in listdir(source_path) if isfile(join(source_path, f)) and ".png" in f]

        for file in files:
            print(join(source_path, file))

            img = Image.open(join(source_path, file))
            npy_arr = numpy.array(img)
            
            if npy_arr.dtype == numpy.uint16:
                npy_arr = npy_arr.view(numpy.int16) + numpy.int16(-32768) 
            elif npy_arr.dtype == numpy.int32:
                npy_arr = (npy_arr - 32768).astype(numpy.int16) 

            content = { 
                "metadata": {
                    "schema": self.SCHEMA_ID,
                    "version": self.VERSION, 
                    "resolution": 1, 
                    "offset": 0, 
                }, 
                "profile_array": npy_arr,
                "intensity_array": None,
            }

            with open(join(destination_path, file.replace(".png", ".gadget3d.pickle")), "wb") as f:
                pickle.dump(content, f, protocol=4)

    def pcd_2_pkl(self, source_path, destination_path, ZResolution = 1, ZOffset = 0):
        import open3d
        files = [f for f in listdir(source_path) if isfile(join(source_path, f)) and ".pcd" in f]

        for file in files:
            print(join(source_path, file))

            pcd = open3d.io.read_point_cloud(join(source_path, file))
            np_arr = numpy.asarray(pcd.points)

            x_len = 0
            for np in np_arr:
                if np[1] != np_arr[0][1]:
                    break
                x_len += 1

            y_len = int(np_arr.shape[0] / x_len)
            
            XOffset = float(np_arr[0][0])
            XResolution = float((np_arr[1][0] - np_arr[0][0]))
            YOffset = float(np_arr[0][1])
            YResolution = float((np_arr[x_len][1] - np_arr[0][1]))

            np_z = numpy.empty((x_len, y_len))
            i = 0
            for y in range(0, y_len):
                for x in range(0, x_len):
                    print(f"({np_arr[i][2]} - {ZOffset}) / {ZResolution}")
                    print(f"({type(np_arr[i][2])} - {type(ZOffset)}) / {type(ZResolution)}")
                    np_z[x][y] = (np_arr[i][2] - ZOffset) / ZResolution
                    i += 1
                
            content = { 
                "metadata": {
                    "schema": self.SCHEMA_ID,
                    "version": self.VERSION, 
                    "resolution": (XResolution, YResolution, ZResolution), 
                    "offset": (XOffset, YOffset, ZOffset), 
                }, 
                "profile_array": np_z.astype(numpy.int16),
                "intensity_array": None,
            }

            with open(join(destination_path, file.replace(".pcd", ".gadget3d.pickle")), "wb") as f:
                pickle.dump(content, f, protocol=4)



if __name__=="__main__":
    # translate=GadgetSurfaceUtils()
    # translate.pkl_2_png("/home/trevor/github/LMI_AI_Solutions/lmi_utils/gadget_utils/src", "/home/trevor/github/LMI_AI_Solutions/lmi_utils/gadget_utils/dest", True)
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument('--option',required=True,help='pkl_2_npy, pkl_2_png, pkl_2_pcd, npy_2_pkl, png_2_pkl, or pcd_2_pkl')
    ap.add_argument('--src',required=True)
    ap.add_argument('--dest',required=True)
    ap.add_argument('--intensity', action='store_true',help='also save intensity image')
    ap.add_argument('--zresolution', default=1, help='ZResolution for PCD to PKL')
    ap.add_argument('--zoffset', default=0, help='ZOffset for PCD to PKL')

    
    args=vars(ap.parse_args())
    option=args['option']
    src=args['src']
    dest=args['dest']
    intensity = args['intensity']
    ZResolution = args['zresolution']
    ZOffset = args['zoffset']

    translate=GadgetSurfaceUtils()

    print(f'Src: {src}')
    print(f'Dest: {dest}')
    
    if not isdir(dest):
        makedirs(dest)

    if option=='pkl_2_npy':
        translate.pkl_2_npy(src,dest,intensity)
    elif option=='pkl_2_png':
        translate.pkl_2_png(src,dest,intensity)
    elif option=='pkl_2_pcd':
        translate.pkl_2_pcd(src,dest)
    elif option=='npy_2_pkl':
        translate.npy_2_pkl(src,dest)
    elif option=='png_2_pkl':
        translate.png_2_pkl(src,dest)
    elif option=='pcd_2_pkl':
        translate.pcd_2_pkl(src,dest,ZResolution,ZOffset)
    else:
        raise Exception('Input option must be pkl_2_npy, pkl_2_png, npy_2_pkl, or png_2_pkl')
