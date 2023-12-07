import shutil
import os
import argparse
import logging
import pandas as pd

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--src', required=True, help='path to source directory. This could be a parent directory of all images, such as each-sku')
    parser.add_argument('-d', '--dest', required=True,  help='path to destination directory')
    parser.add_argument('-c', '--csv', required=True,  help='csv file with: fname,mean,max,std')
    parser.add_argument('-t', '--threshold', type=float, default=0.0,  help='threshold for anomaly detection scores')
    
    args = parser.parse_args()

    df=pd.read_csv(args.csv)

    # sorted_df=df.sort_values(by=['max'],ascending=False)
    
    if not os.path.exists(args.dest):
        os.makedirs(args.dest)
    
    for root, dirs, files in os.walk(args.src):
        for file in files:
            row_id=df[df['fname']==file.replace('_annot.png','.png')].index.tolist()
            if row_id:
                if len(row_id)>1:
                    logger.info(f'Found multiple entries at row: {row_id}. Reducing to {row_id[0]}')
                row_id=row_id[0]
                ad_max=df.iloc[row_id]['mean']
                ad_max_str=f"{ad_max:006.2f}"
                
                # skip if below threshold
                if ad_max<args.threshold:
                    continue
                
                # support both the original and the annotated images
                fname=df.iloc[row_id]['fname']
                if not os.path.isfile(os.path.join(root, fname)):
                    fname = fname.replace('.png','_annot.png')
                    
                path = os.path.join(root, fname)
                fname_modified=ad_max_str+"-"+fname
                
                # skip if already exists
                dest=os.path.join(args.dest, fname_modified)
                if os.path.isfile(dest):
                    logger.info(f'{fname_modified} already exists in {args.dest}, skip')
                    continue
                
                logger.info(f'Copy {path} to {args.dest}')
                shutil.copy(path, dest)
