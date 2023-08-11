source /repos/LMI_AI_Solutions/lmi_ai.env

export LD_PRELOAD=/lib/aarch64-linux-gnu/libGLdispatch.so.0

# convert to tensorRT engine
python3 -m yolov5.export --weights $MODEL_PATH/best.pt \
    --imgsz $IM_H $IM_W  --include engine --half --device 0

# validation
if [ -d /app/images ]; then
    python3 -m yolov5_lmi.trt.infer_trt --engine $MODEL_PATH/best.engine \
        --imsz $IM_H $IM_W --path_imgs /app/images --path_out /app/validation
else
    echo "Testing images are not found in /app/images, skip validation"
fi
