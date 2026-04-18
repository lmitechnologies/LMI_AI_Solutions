#!/bin/bash
if [ "$#" -eq 0 ]; then
    echo "Usage: $0 <argument>"
    exit 1
fi

outpath=tests/outputs
ARGUMENT=$1
if [ "$ARGUMENT" == "v1-all" ]; then
    pytest --html=$outpath/lmi_utils.html tests/lmi_utils/
    pytest --html=$outpath/object_detectors.html tests/object_detectors/
    pytest --html=$outpath/classifiers.html tests/classifiers/
    pytest --html=$outpath/anomaly_detectors_v1.html tests/anomaly_detectors/anomalib_lmi/test_v1.py
elif [ "$ARGUMENT" == "od" ]; then
    pytest --html=$outpath/object_detectors.html tests/object_detectors/
elif [ "$ARGUMENT" == "utils" ]; then
    pytest --html=$outpath/lmi_utils.html tests/lmi_utils/ tests/lmi_common
elif [ "$ARGUMENT" == "cls" ]; then
    pytest --html=$outpath/classifiers.html tests/classifiers/
elif [ "$ARGUMENT" == "ad-v1" ]; then
    pytest --html=$outpath/anomaly_detectors_v1.html tests/anomaly_detectors/anomalib_lmi/test_v1.py
elif [ "$ARGUMENT" == "ad-v2" ]; then
    pytest --html=$outpath/anomaly_detectors_v2.html tests/anomaly_detectors/anomalib_lmi/test_v2.py
else
    echo "Invalid argument. Please use 'v1-all' 'od' 'utils' 'ad-v1' 'ad-v2' 'cls'. "
    exit 1
fi
