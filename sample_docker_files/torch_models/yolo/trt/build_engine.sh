# Compile tensorrtx

# Errors if it's in the dockerfile - Could NOT find CUDA (missing: CUDA_CUDART_LIBRARY) (found version "10.2")
# export CUDA_CUDART_LIBRARY=/usr/local/cuda/lib64/stubs
# mkdir -p /app/build_trt && cd /app/build_trt && cmake /app/tensorrtx/yolov5 && make

WEIGHT_PATH=/app/best.pt

python3 -m yolov5.export --weights /app/best.pt --imgsz $IM_H $IM_W  --include engine --half --device 0

CONVERSION_NAME="$(date +'%Y-%m-%d-%H-%M')"
OUTPUT_PATH=/app/trt_engines/$CONVERSION_NAME
echo OUTPUT_PATH $OUTPUT_PATH
mkdir -p $OUTPUT_PATH

cp best.engine $OUTPUT_PATH/


