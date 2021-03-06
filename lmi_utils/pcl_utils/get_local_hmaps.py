import csv
import argparse
import pandas as pd
import os
import pcl_utils.point_cloud as pcloud

''' MODULE:
        get_local_hmaps.py

    USAGE:
        python3 get_local_hmaps.py --csv_in /my/csv/path/my_file.csv --data_path /my/data/path/ --zmin -100 --zmax 100 --extra_str remove_this_part

    DESCRIPTION:
        Generates new local pointcloud surface (.png) from data file and csv containing bounding boxes.  The .csv file is generated by function: 
            fringeai_ml/object_detection/TensorFlow/utils/via_json_to_csv.py when used with images labeled with bounding boxes.  Currently does not support
            .csv files created from labeling data that includes mask regions.
'''

def gen_surface_and_save(pcd,x_ul,x_lr,y_ul,y_lr,color_option,zmin,zmax,region_id,label,output_path,pcd_file):
    '''
    Description:
        -Prunes background from the foreground that is defined by the input bounding box.  Then creates a new heightmap and saves as .png.

    Arguments:
        pcd: pointcloud object from point_cloud.py
        x_ul: bounding box upper left x coordinate
        y_ul: bounding box upper left y coordinate
        x_lr: bounding box lower right x coordinate
        y_lr: bounding box lower right y coordinate
        color_option: final .png image format
        label: label for foreground region
        output_path: directory for output .png files
        pcd_file: corresponding .pcd file that includes the background and foreground

    Calls:
        pcd.convert_points_to_grayscale_image or pcd.convert_points_to_color_image to encode height map
        pcd.save_img() to write the image
        pcd.reinitialize_fp_image to reinitialize the base heightmap as floating point
    
    Returns:
        None

    Raises:
        General exception for invalid color choice
    '''
    pcd.prune(x_ul,x_lr,y_ul,y_lr,zmin,zmax)
    if color_option=='gray':
        pcd.convert_points_to_grayscale_image(normalize=False,ZMIN=zmin,ZMAX=zmax,contrast_enhancement=True)
    elif color_option=='rgb':
        pcd.convert_points_to_color_image(normalize=False,rgb_mapping='rainbow',ZMIN=zmin,ZMAX=zmax,contrast_enhancement=True)
    else:
        raise Exception('Invalid Color Option')
    suffix='_region_'+str(region_id)+'_'+label
    
    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    outfile=os.path.split(pcd_file)[1]
    outfile=os.path.splitext(outfile)[0]
    outfile_path=os.path.join(output_path,outfile)+suffix+'.png'


    pcd.save_img(outfile_path)
    pcd.reinitialize_fp_image()


def prune_ROIs(csv_file_path,data_path,output_path,zmin=-100,zmax=100,extra_str=None,color_option='gray'):
    '''
    Description:
        -Extracts filename and bounding box information from CSV files.

    Arguments:
        csv_file_path: input file path
        data_path: source data file
        output_path: directory for output images
        zmin: lower bound for height mapping
        zmax: upper bound for height mapping
        extra_str: input image suffix that, when removed, gives the .pcd file name
        color_option: rgb or grayscale mapping

    Calls:
        PointCloud() from point_cloud.py to store and process point cloud data
        gen_surface_and_save() to crop volume in bounding box and save as new image

    Returns:
        None

    Raises:
        General exception for invalid syntax
    '''
    
    # Path contains all first level point clouds
    df=pd.read_csv(csv_file_path,sep=';',names=['Path','Label','Shape','Position','X','Y'])

    files=df['Path'].unique()
    
    pcd=pcloud.PointCloud()
    # step through csv file processing each bounding box
    for each_file in files:
        df_file=df[df['Path']==each_file]
        img_file=os.path.splitext(each_file)[0]
        # remove extra strings added to point cloud base name 
        if extra_str is not None:
            ind=img_file.find(extra_str)
        else:
            ind=len(img_file)
        # get the point cloud
        pcd_file=data_path+'/'+img_file[0:ind]+'.pcd'
        try: 
            pcd.read_points(pcd_file)
        except:
            print('[Error] Could not open .pcd file: ',pcd_file)
        region_id=0
        # check bounding box shape
        bb_shape_check=[False,False]
        for row in df_file.itertuples():
            label=row.Label
            if row.Shape != "rect":
                #TODO: support masks
                raise Exception('Only bounding boxes are supported.')
            #
            if row.Position == 'upper left':
                x_ul=row.X
                y_ul=row.Y
                bb_shape_check[0]=True
                if bb_shape_check[0]+bb_shape_check[1]==2:
                    gen_surface_and_save(pcd,x_ul,x_lr,y_ul,y_lr,color_option,zmin,zmax,region_id,label,output_path,pcd_file)
                    bb_shape_check=[False,False]
                    region_id+=1
            elif row.Position == 'lower right':
                x_lr=row.X
                y_lr=row.Y
                bb_shape_check[1]=True
                if bb_shape_check[0]+bb_shape_check[1]==2:
                    gen_surface_and_save(pcd,x_ul,x_lr,y_ul,y_lr,color_option,zmin,zmax,region_id,label,output_path,pcd_file)
                    bb_shape_check=[False,False]
                    region_id+=1
            else:
                raise Exception(f'Position variable: {row.Position} is not supported.')
        

if __name__ == "__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument('-c','--csv_in',required=True,help='csv input file with fname, label, bounding box')
    ap.add_argument('-d','--data_path',required=True,help='data path')
    ap.add_argument('--output_path',required=True,help='output path')
    ap.add_argument('--zmin',type=float,default=-100,help='minimum z')
    ap.add_argument('--zmax',type=float,default=100,help='maximum z')
    ap.add_argument('-x','--extra_str',default=None,help='extra string appended to image file used to locate point cloud.')
    ap.add_argument('--color_option',default='gray',help='Color image option: gray or rgb')

    args=vars(ap.parse_args())
    
    prune_ROIs(args['csv_in'],args['data_path'],args['output_path'],args['zmin'],args['zmax'],args['extra_str'],args['color_option'])
