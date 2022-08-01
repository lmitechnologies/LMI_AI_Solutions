import argparse
import numpy as np
import shutil
import os
import csv

def filter_by_label(inputfile,datapath,outpath):
    files=[]
    with open(inputfile) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';')
        for row in csv_reader:
            files.append(row[0])
    files_np=np.array(files)
    files_unique=list(np.unique(files_np))

    for f in files_unique:
        fsrc=os.path.join(datapath,f)
        fdst=os.path.join(outpath,f)
        shutil.copyfile(fsrc,fdst)
        print(f'Copied file: {f}')



if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--input_csv_file',required=True)
    ap.add_argument('--data_path',required=True)
    ap.add_argument('--out_path',required=True)

    args=vars(ap.parse_args())

    inputfile=args['input_csv_file']
    datapath=args['data_path']
    outpath=args['out_path']

    filter_by_label(inputfile, datapath, outpath)
