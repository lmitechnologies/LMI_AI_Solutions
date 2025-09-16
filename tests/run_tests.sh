#!/bin/bash
if [ "$#" -eq 0 ]; then
    echo "Usage: $0 <argument>"
    exit 1
fi

INSTALL=$2
if [ "$INSTALL" == "install-packages" ]; then
    echo "Installing packages"
    pip3 install -e lmi_utils/
    pip3 install -e object_detectors/
    pip3 install -e anomaly_detectors/
    pip3 install -e classifiers/
else
    echo "using repo env"
    source /app/repo/lmi_ai.env
fi

# Git LFS pull all the artifacts
echo "running git lfs pull"
git lfs pull
echo "git lfs pull complete"

outpath=tests/outputs
ARGUMENT=$1
if [ "$ARGUMENT" == "v1-all" ]; then
    pytest --html=$outpath/lmi_utils_v1.html tests/lmi_utils/
    pytest --html=$outpath/object_detectors_v1.html tests/object_detectors/
    pytest --html=$outpath/classifiers_v1.html tests/classifiers/
    pytest --html=$outpath/anomaly_detectors_v1.html --ignore=tests/anomaly_detectors/anomalib_lmi/test_anomaly_model.py tests/anomaly_detectors
elif [ "$ARGUMENT" == "v0-all" ]; then
    pytest --html=$outpath/lmi_utils_v0.html tests/lmi_utils/
    pytest --html=$outpath/object_detectors_v0.html tests/object_detectors/
    pytest --html=$outpath/classifiers_v0.html tests/classifiers/
    pytest --html=$outpath/anomaly_detectors_v0.html --ignore=tests/anomaly_detectors/anomalib_lmi/test_anomaly_model2.py tests/anomaly_detectors
elif [ "$ARGUMENT" == "od" ]; then
    pytest --html=$outpath/object_detectors.html tests/object_detectors/
elif [ "$ARGUMENT" == "utils" ]; then
    pytest --html=$outpath/lmi_utils.html tests/lmi_utils/
elif [ "$ARGUMENT" == "cls" ]; then
    pytest --html=$outpath/classifiers.html tests/classifiers/
elif [ "$ARGUMENT" == "ad-v0" ]; then
    pytest --html=$outpath/anomaly_detectors_v0.html --ignore=tests/anomaly_detectors/anomalib_lmi/test_anomaly_model2.py tests/anomaly_detectors
elif [ "$ARGUMENT" == "ad-v1" ]; then
    pytest --html=$outpath/anomaly_detectors_v1.html --ignore=tests/anomaly_detectors/anomalib_lmi/test_anomaly_model.py tests/anomaly_detectors
else
    echo "Invalid argument. Please use 'v1-all' 'v0-all' 'od' 'utils' 'ad-v0' 'ad-v1' 'cls'. "
    exit 1
fi
