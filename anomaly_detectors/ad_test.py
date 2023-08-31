






if __name__=='__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('-i','--input_path', required=True, help='the path to images')
    ap.add_argument('-o','--output_path', default='./')
    ap.add_argument('--width', type=int, default=None)
    ap.add_argument('--height',type=int, default=None)
    args = vars(ap.parse_args())