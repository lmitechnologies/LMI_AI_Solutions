from PIL import Image
import pickle
import numpy
from os import listdir, makedirs
from os.path import isfile, join, isdir

TWO_TO_FIFTEEN=32768

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
                
           
    @staticmethod     
    def convert_to_xyz(profile, resolution, offset, img_intensity=None):
        np_z=[]
        np_x=[]
        np_y=[]
        intensity=[]
        i = 0
        
        # convert to int16
        # the metadata only works with int16
        if profile.dtype == numpy.uint16:
            profile = profile.view(numpy.int16) + numpy.int16(-TWO_TO_FIFTEEN)

        shape = profile.shape
        for y in range(shape[0]):
            for x in range(shape[1]):
                if profile[y][x] != -1*TWO_TO_FIFTEEN:
                    np_x.append(offset[0] + x * resolution[0])
                    np_y.append(offset[1] + y * resolution[1])
                    np_z.append(offset[2] + profile[y][x] * resolution[2])
                    if img_intensity is not None:
                        intensity.append(img_intensity[y][x]/255.0)
                    i += 1
        
        np_points = numpy.empty((i, 3))
        np_points[:, 0] = numpy.array(np_x)
        np_points[:, 1] = numpy.array(np_y)
        np_points[:, 2] = numpy.array(np_z)
        np_intensity=numpy.array(intensity)
        return np_points,np_intensity
        
                
    def pkl_2_pcd(self, source_path, destination_path,source_path_intensity=None):
        import open3d
        files = [f for f in listdir(source_path) if isfile(join(source_path, f)) and ".gadget3d.pickle" in f]
        
        use_intensity=True if source_path_intensity is not None else False

        for file in files:
            print(join(source_path, file))

            with open(join(source_path, file), "rb") as f:
                content = pickle.load(f)
            
            if use_intensity:
                try:
                    fname_intensity=file.replace(".gadget3d.pickle", ".gadget2d.jpg")
                    path_intensity=join(source_path_intensity,fname_intensity)
                    print(f'[INFO] Loading intensity image from:{path_intensity}')
                    img_intensity=Image.open(path_intensity)
                    img_intensity=img_intensity.convert('RGB') #convert to color
                    img_intensity=numpy.array(img_intensity).astype(numpy.float32)
                except:
                    print(f'[WARNING] Failed to load intensity image.')
                    use_intensity=False

            profile = content["profile_array"]
            resolution = content["metadata"]["resolution"]
            offset = content["metadata"]["offset"]

            # convert to 3d points
            if use_intensity:
                np_points,np_intensity = self.convert_to_xyz(profile, resolution, offset, img_intensity)
            else:
                np_points = self.convert_to_xyz(profile, resolution, offset)
            
            pcd = open3d.geometry.PointCloud()
            pcd.points = open3d.utility.Vector3dVector(np_points)
            if use_intensity:
                pcd.colors = open3d.utility.Vector3dVector(np_intensity)
            open3d.io.write_point_cloud(join(destination_path, file.replace(".gadget3d.pickle", ".pcd")), pcd)
    
    def tar_2_pcd(self, source_path, destination_path, source_path_intensity=None):
        import open3d
        import tarfile
        import json

        files = [f for f in listdir(source_path) if isfile(join(source_path, f)) and ".gadget3d.tar" in f]

        use_intensity=True if source_path_intensity is not None else False
        
        for file in files:
            print(join(source_path, file))
            
            with tarfile.open(join(source_path, file), "r") as tar:
                dest = join(destination_path  , file.replace(".gadget3d.tar", ""))
                tar.extractall(dest)

                png = Image.open(join(dest, "profile.png"))
                if use_intensity:
                    try:
                        fname_intensity=file.replace(".gadget3d.tar", ".gadget2d.jpg")
                        path_intensity=join(source_path_intensity,fname_intensity)
                        print(f'[INFO] Loading intensity image from:{path_intensity}')
                        img_intensity=Image.open(path_intensity)
                        img_intensity=img_intensity.convert('RGB') #convert to color
                        img_intensity=numpy.array(img_intensity).astype(numpy.float32)
                    except:
                        print(f'[WARNING] Failed to load intensity image.')
                        use_intensity=False
                        
                metadata = None
                with open(join(dest, "metadata.json"), "r") as f:
                    metadata = json.load(f)

                profile = numpy.array(png)
                resolution = metadata["resolution"]
                offset = metadata["offset"]

                # convert to 3d points
                if use_intensity:
                    np_points,np_intensity = self.convert_to_xyz(profile, resolution, offset, img_intensity)
                else:
                    np_points = self.convert_to_xyz(profile, resolution, offset)
                
                pcd = open3d.geometry.PointCloud()
                pcd.points = open3d.utility.Vector3dVector(np_points)
                if use_intensity:
                    pcd.colors = open3d.utility.Vector3dVector(np_intensity)
                open3d.io.write_point_cloud(join(destination_path, file.replace(".gadget3d.tar", ".pcd")), pcd)


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
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument('--option',required=True,help='pkl_2_npy, pkl_2_png, pkl_2_pcd, npy_2_pkl, png_2_pkl, tar_2_pcd, or pcd_2_pkl')
    ap.add_argument('--src',required=True)
    ap.add_argument('--dest',required=True)
    ap.add_argument('--src_intensity',default=None,help='Intensity image path if converting tar to pcd w/ intensity.')
    ap.add_argument('--intensity', action='store_true',help='also save intensity image')
    ap.add_argument('--zresolution', help='ZResolution for PCD to PKL')
    ap.add_argument('--zoffset', help='ZOffset for PCD to PKL')

    
    args=vars(ap.parse_args())
    option=args['option']
    src=args['src']
    dest=args['dest']
    src_intensity=args['src_intensity']
    intensity = args['intensity']

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
    elif option=='tar_2_pcd':
        translate.tar_2_pcd(src,dest,source_path_intensity=src_intensity)
    elif option=='pcd_2_pkl':
        ZResolution = args['zresolution']
        ZOffset = args['zoffset']
        translate.pcd_2_pkl(src,dest,ZResolution,ZOffset)
    else:
        raise Exception('Input option must be pkl_2_npy, pkl_2_png, npy_2_pkl, or png_2_pkl')
