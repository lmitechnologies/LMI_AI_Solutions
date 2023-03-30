import argparse
import sys
sys.path.append('/app/LMI_AI_Solutions/lmi_utils')
sys.path.append('/app/LMI_AI_Solutions/anomaly_detectors')

from padim.padim import PaDiM


if __name__=='__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--input_path', '-i', required=True, help='the path to the tf "saved_model"')
    ap.add_argument('--out_path', '-o', required=True, help='the output path of trt engines')
    ap.add_argument('--calibration_path', '-c', required=True, help='the path to calibration data')
    ap.add_argument('--gpu_mem', default=4096, type=int, help='the gpu memory allocation, default to 4096')
    args = ap.parse_args()

    padim = PaDiM(GPU_memory=args.gpu_mem)
    padim.convert_tensorRT(args.input_path, args.out_path, args.calibration_path)
