#!/bin/bash
set -e
if [ "$#" -eq 0 ]; then
    echo "Usage: $0 <argument>"
    exit 1
fi

outpath=tests/outputs
ARGUMENT=$1

# Build any missing TensorRT engines for the given backends before their tests run. Engines are
# gitignored (platform-specific), so this regenerates them locally. No-op without a GPU/TensorRT,
# and per-backend build failures are warnings — the corresponding TRT tests then skip.
build_engines() {
    python -m tests.build_test_engines --backend "$1" --skip-existing --if-available
}

if [ "$ARGUMENT" == "all-v1" ]; then
    build_engines rf_detr,detectron2,ad_v1
    pytest --html=$outpath/all-v1.html tests/lmi_utils tests/lmi_common tests/object_detectors tests/classifiers tests/anomaly_detectors/anomalib_lmi/test_v1.py
elif [ "$ARGUMENT" == "od" ]; then
    build_engines rf_detr,detectron2
    pytest --html=$outpath/object_detectors.html tests/object_detectors
elif [ "$ARGUMENT" == "utils" ]; then
    pytest --html=$outpath/lmi_utils.html tests/lmi_utils
    pytest --html=$outpath/lmi_common.html tests/lmi_common
elif [ "$ARGUMENT" == "cls" ]; then
    pytest --html=$outpath/classifiers.html tests/classifiers
elif [ "$ARGUMENT" == "ad-v1" ]; then
    build_engines ad_v1
    pytest --html=$outpath/anomaly_detectors_v1.html tests/anomaly_detectors/anomalib_lmi/test_v1.py
elif [ "$ARGUMENT" == "ad-v2" ]; then
    build_engines ad_v2
    pytest --html=$outpath/anomaly_detectors_v2.html tests/anomaly_detectors/anomalib_lmi/test_v2.py tests/anomaly_detectors/anomalib_lmi/memory_estimation/
else
    echo "Invalid argument. Please use 'all-v1' 'od' 'utils' 'ad-v1' 'ad-v2' 'cls'. "
    exit 1
fi
