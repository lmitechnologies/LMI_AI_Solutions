#!/bin/bash

action=$1
model=$2
onnx_dir=/app/onnx
engine_dir=/app/engine

if [[ -z $model ]]; then
    model=padim
fi

echo action $action, model $model

trained_models_dir=/app/out/results/$model/trained-models
engines_dir=/app/out/engines/$model

if [[ ! -f $onnx_dir/model.onnx ]]; then
    onnx_dir=$trained_models_dir
fi
if [[ ! -f $engine_dir/model.engine ]]; then
    engine_dir=$engines_dir
fi

if [ "$action" = "train" ] || [ "$action" = "all" ]; then

    # -O for disabling assert in the python script to workaround following error:
    # https://github.com/openvinotoolkit/anomalib/issues/1238
    python3 -O anomalib/tools/train.py --config /app/ws/configs/$model.yaml

    build_name="$(date +'%Y-%m-%d-%H-%M')"
    mkdir -p $trained_models_dir && \
    mv /app/out/results/$model/model/train/run/weights/onnx $trained_models_dir/$build_name

    # tune hyp parameters
    # python anomalib/tools/hpo/sweep.py --model $model --model_config /app/configs/config.yaml \
    #     --sweep_config tools/hpo/configs/comet.yaml
fi

python3 ws/anomaly_model.py $action $model $onnx_dir $engine_dir