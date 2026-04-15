#!/bin/bash
if [ "$#" -eq 0 ]; then
    echo "Usage: $0 <argument>"
    exit 1
fi

# Git LFS pull all the artifacts
echo "current directory: $(pwd)"
git config --global --add safe.directory "$(pwd)"
echo "running git lfs pull"
git lfs pull
if [ $? -ne 0 ]; then
    echo "git lfs pull failed"
    exit 1
fi
echo "git lfs pull complete"

# install the mounted ais packages
pip install -e .

outpath=tests/outputs
ARGUMENT=$1
if [ "$ARGUMENT" == "v1-all" ]; then
    pytest --html=$outpath/lmi_utils.html tests/lmi_utils/
    pytest --html=$outpath/object_detectors.html tests/object_detectors/
    pytest --html=$outpath/classifiers.html tests/classifiers/
    pytest --html=$outpath/anomaly_detectors_v1.html tests/anomaly_detectors/anomalib_lmi/test_anomaly_model_v1.py
elif [ "$ARGUMENT" == "od" ]; then
    pytest --html=$outpath/object_detectors.html tests/object_detectors/
elif [ "$ARGUMENT" == "utils" ]; then
    pytest --html=$outpath/lmi_utils.html tests/lmi_utils/
elif [ "$ARGUMENT" == "cls" ]; then
    pytest --html=$outpath/classifiers.html tests/classifiers/
elif [ "$ARGUMENT" == "ad-v1" ]; then
    pytest --html=$outpath/anomaly_detectors_v1.html tests/anomaly_detectors/anomalib_lmi/test_anomaly_model_v1.py
elif [ "$ARGUMENT" == "ad-v2" ]; then
    pytest --html=$outpath/anomaly_detectors_v2.html tests/anomaly_detectors/anomalib_lmi/test_anomaly_model_v2.py
else
    echo "Invalid argument. Please use 'v1-all' 'od' 'utils' 'ad-v0' 'ad-v1' 'ad-v2' 'cls'. "
    exit 1
fi
