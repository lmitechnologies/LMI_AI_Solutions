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

    # -O for disabling assert in the python script to workaround following error:
    # https://github.com/openvinotoolkit/anomalib/issues/1238
    python3 -O anomalib/tools/train.py --config /app/ws/configs/$MODEL.yaml

    build_name="$(date +'%Y-%m-%d-%H-%M')"
    mkdir -p $trained_models_dir && \
    mv /app/out/results/$MODEL/model/train/run/weights/onnx $trained_models_dir/$build_name

    # tune hyp parameters
    # python anomalib/tools/hpo/sweep.py --model $MODEL --model_config /app/configs/config.yaml \
    #     --sweep_config tools/hpo/configs/comet.yaml
fi

python3 ws/anomaly_model.py $action $MODEL $onnx_dir $engine_dir