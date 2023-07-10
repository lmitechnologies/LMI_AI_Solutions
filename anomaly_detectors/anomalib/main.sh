#!/bin/bash

if [ "$ACTION" = "train" ]; then
    # ensure generated model is named model.ckpt rather than model-N.ckpt
    rm -rf /app/out/results/$MODEL/model/run/weights/lightning/

    python3 anomalib/tools/train.py --config /app/configs/$MODEL.yaml

    build_name="$(date +'%Y-%m-%d-%H-%M')"
    python3 anomalib/tools/inference/lightning_inference.py \
        --config /app/configs/$MODEL.yaml \
        --weights /app/out/results/$MODEL/model/run/weights/lightning/model.ckpt \
        --input /app/data/top/test/defects \
        --output /app/out/evaluation/$MODEL/$build_name

    # tune hyp parameters
    # python anomalib/tools/hpo/sweep.py \
    #     --model $MODEL --model_config /app/configs/config.yaml \
    #     --sweep_config tools/hpo/configs/comet.yaml
else
    python3 anomaly_model.py $ACTION $MODEL
fi
