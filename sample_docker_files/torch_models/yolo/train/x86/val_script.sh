source /app/LMI_AI_Solutions/lmi_ai.env
python3 -m yolov5.detect --source /app/test/data/640x640 --img 640 --weights /app/trained-inference-models/$TRAIN_DIR/best.pt --project /app/test/ --name $TRAIN_DIR --save-csv
