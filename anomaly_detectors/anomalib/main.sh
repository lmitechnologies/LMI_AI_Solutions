#!/bin/bash

if [ "$ACTION" = "train" ] || [ "$ACTION" = "all" ]; then
    # ensure generated model is named model.ckpt rather than model-N.ckpt
    rm -rf /app/out/results/$MODEL/model/run/weights/
    python3 anomalib/tools/train.py --config /app/configs/$MODEL.yaml

    build_name="$(date +'%Y-%m-%d-%H-%M')"
    if [[ $TEST_IMAGES ]]; then
        python3 anomalib/tools/inference/lightning_inference.py \
            --config /app/configs/$MODEL.yaml \
            --weights /app/out/results/$MODEL/model/run/weights/lightning/model.ckpt \
            --input /app/data/$TEST_IMAGES \
            --output /app/out/evaluation/$MODEL/$TEST_IMAGES
    fi

    trained_model_out_dir=/app/out/results/$MODEL/model/trained-models/$build_name
    mkdir -p $trained_model_out_dir && \
    cp -r /app/out/results/$MODEL/model/run/weights/* $trained_model_out_dir

    # tune hyp parameters
    # python anomalib/tools/hpo/sweep.py \
    #     --model $MODEL --model_config /app/configs/config.yaml \
    #     --sweep_config tools/hpo/configs/comet.yaml
fi

python3 anomaly_model.py $ACTION $MODEL