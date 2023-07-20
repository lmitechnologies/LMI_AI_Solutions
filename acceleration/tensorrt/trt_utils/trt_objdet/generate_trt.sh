python3 trt_objdet.py   --input_saved_model_dir /app/trained-inference-models/faster_rcnn_resnet50_v1_640x640_coco17_tpu-8/saved_model \
                        --data_dir /app/data/coco/val2017 \
                        --batch_size 1 \
                        --use_tftrt \
                        --precision FP16 \
                        --total_max_samples 100 \
                        --model_name Faster \
                        --model_source Coco \
                        --max_workspace_size 2147483648 \
                        --gpu_mem_cap 3000 \
                        --output_tensors_name detection_boxes,detection_classes,detection_scores,num_detections \
                        --num_warmup_iterations 1 \
                        --output_saved_model_dir /app/trained-inference-models/faster_rcnn_resnet50_v1_640x640_coco17_tpu-8/saved_model \
                        --input_size 640