source /app/LMI_AI_Solutions/lmi_ai.env

python3 -m tf_objdet.lmi_utils.exporter_main_v2 --input_type image_tensor \
    --pipeline_config_path /app/trained-inference-models/fasterrcnn_400x400/2021-12-08_400/pipeline.config \
    --trained_checkpoint_dir /app/trained-inference-models/fasterrcnn_400x400/2021-12-08_400/checkpoint  \
    --output_directory /app/trained-inference-models/fasterrcnn_400x400/2021-12-08_400_rebuild_tf230