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
fi
ARGUMENT=$1
if [ "$ARGUMENT" == "v1-all" ]; then
    pytest --test-package=True --html=/app/repo/tests/lmi_utils_v1_packaged.html tests/lmi_utils/
    pytest --test-package=True --html=/app/repo/tests/object_detectors_v1_packaged.html tests/object_detectors/
    pytest --test-package=True --html=/app/repo/tests/anomaly_detectors_v1_packaged.html tests/anomaly_detectors/anomalib_lmi/test_anomaly_model2*
    pytest --test-package=False --html=/app/repo/tests/lmi_utils_v1.html tests/lmi_utils/
    pytest --test-package=False --html=/app/repo/tests/object_detectors_v1.html tests/object_detectors/
    pytest --test-package=False --html=/app/repo/tests/anomaly_detectors_v1.html tests/anomaly_detectors/anomalib_lmi/test_anomaly_model2*
    exit 0
elif [ "$ARGUMENT" == "v0-all" ]; then
    pytest --test-package=True --html=/app/repo/tests/lmi_utils_v0_packaged.html tests/lmi_utils/
    pytest --test-package=True --html=/app/repo/tests/object_detectors_v0_packaged.html tests/object_detectors/
    pytest --test-package=True --html=/app/repo/tests/anomaly_detectors_v0_packaged.html tests/anomaly_detectors/anomalib_lmi/test_anomaly_model.py
    pytest --test-package=False --html=/app/repo/tests/lmi_utils_v0.html tests/lmi_utils/
    pytest --test-package=False --html=/app/repo/tests/object_detectors_v0.html tests/object_detectors/
    pytest --test-package=False --html=/app/repo/tests/anomaly_detectors_v0.html tests/anomaly_detectors/anomalib_lmi/test_anomaly_model.py
    exit 0
elif [ "$ARGUMENT" == "object_detectors" ]; then
    pytest --test-package=True --html=/app/repo/tests/object_detectors_packaged.html tests/object_detectors/
    pytest --test-package=False --html=/app/repo/tests/object_detectors.html tests/object_detectors/
    exit 0
elif [ "$ARGUMENT" == "lmi_utils" ]; then
    pytest --test-package=True --html=/app/repo/tests/lmi_utils_packaged.html tests/lmi_utils/
    pytest --test-package=False --html=/app/repo/tests/lmi_utils.html tests/lmi_utils/
    exit 0
elif [ "$ARGUMENT" == "anomaly_detectors-v0" ]; then
    pytest --test-package=True --html=/app/repo/tests/anomaly_detectors_v0_packaged.html tests/anomaly_detectors/anomalib_lmi/test_anomaly_model.py
    pytest --test-package=False --html=/app/repo/tests/anomaly_detectors_v0.html tests/anomaly_detectors/anomalib_lmi/test_anomaly_model.py
    exit 0
elif [ "$ARGUMENT" == "anomaly_detectors-v1" ]; then
    pytest --test-package=True --html=/app/repo/tests/anomaly_detectors_v1_packaged.html tests/anomaly_detectors/anomalib_lmi/test_anomaly_model2.py
    pytest --test-package=False --html=/app/repo/tests/anomaly_detectors_v1.html tests/anomaly_detectors/anomalib_lmi/test_anomaly_model2.py
    exit 0
fi
echo "Invalid argument. Please use 'v1-all' 'v0-all' 'object_detectors' 'lmi_utils' 'anomaly_detectors-v0' 'anomaly_detectors-v1'. "