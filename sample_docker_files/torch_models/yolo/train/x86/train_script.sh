source /app/LMI_AI_Solutions/lmi_ai.env
python3 -m yolov5.train --img 640 --batch $BATCH --epoch $EPOCH --data /app/yolo.yaml --weights yolov5s.pt --project /app/training --name $TRAIN_DIR --exist-ok
mkdir -p /app/trained-inference-models/$TRAIN_DIR
cp /app/training/weights/best.pt /app/trained-inference-models/$TRAIN_DIR
cp /app/training/weights/last.pt /app/trained-inference-models/$TRAIN_DIR