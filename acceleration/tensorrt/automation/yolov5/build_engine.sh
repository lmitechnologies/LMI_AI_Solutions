# Compile tensorrtx

# Errors if it's in the dockerfile - Could NOT find CUDA (missing: CUDA_CUDART_LIBRARY) (found version "10.2")
export CUDA_CUDART_LIBRARY=/usr/local/cuda/lib64/stubs
mkdir -p /app/build_trt && cd /app/build_trt && cmake /app/tensorrtx/yolov5 && make

CONVERSION_NAME="$(date +'%Y-%m-%d-%H-%M')"
OUTPUT_PATH=/app/trt_engines/$CONVERSION_NAME
mkdir -p $OUTPUT_PATH

echo WEIGHT_PATH $WEIGHT_PATH

OUT_PREFIX=model
WEIGHT_PATH=/app/weight.pt
CONFIG_PATH=/app/config.yaml

# generate weights
python3 /app/LMI_AI_Solutions/object_detectors/yolov5/gen_wts.py -w "$WEIGHT_PATH" -o "$OUTPUT_PATH"/"$OUT_PREFIX".wts

# build engine
/app/build_trt/yolov5 -c "$CONFIG_PATH" -w "$OUTPUT_PATH"/$OUT_PREFIX.wts -o "$OUTPUT_PATH"/model.engine
# copy production shared object
cp /app/build_trt/libmyplugins.so $OUTPUT_PATH/libmyplugins.so

