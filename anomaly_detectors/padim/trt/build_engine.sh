source /repos/LMI_AI_Solutions/lmi_ai.env

python -m padim.trt.convert_trt -i $MODEL_PATH -o $TRT_PATH -c $CALIB_PATH
