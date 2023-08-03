#!/bin/bash

action=$1
MODEL=$2
onnx_dir=/app/mounted/onnx
engine_dir=/app/mounted/engine

trained_models_dir=/app/out/results/$MODEL/trained-models
engines_dir=/app/out/engines/$MODEL

if [[ ! -d $onnx_dir ]]; then
    onnx_dir=$trained_models_dir
fi
if [[ ! -d $engine_dir ]]; then
    engine_dir=$engines_dir
fi

if [ "$action" = "train" ] || [ "$action" = "all" ]; then
    # ensure generated model is named model.ckpt rather than model-N.ckpt
    rm -rf /app/out/results/$MODEL/model/run/weights/

    python3 anomalib/tools/train.py --config /app/ws/configs/$MODEL.yaml

    build_name="$(date +'%Y-%m-%d-%H-%M')"
    mkdir -p $trained_models_dir && \
    mv /app/out/results/$MODEL/model/run/weights/onnx $trained_models_dir/$build_name

    test_data_dir=/app/data/test
    if [[ -d $test_data_dir ]]; then
        python3 anomalib/tools/inference/lightning_inference.py \
            --config /app/ws/configs/$MODEL.yaml \
            --weights /app/out/results/$MODEL/model/run/weights/lightning/model.ckpt \
            --input $test_data_dir \
            --output /app/out/evaluation/$MODEL/$build_name
    fi

    # tune hyp parameters
    # python anomalib/tools/hpo/sweep.py --model $MODEL --model_config /app/configs/config.yaml \
    #     --sweep_config tools/hpo/configs/comet.yaml
fi

python3 ws/anomaly_model.py $action $MODEL $onnx_dir $engine_dir