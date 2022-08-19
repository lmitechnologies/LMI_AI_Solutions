docker run -it --rm    --gpus="all"    --shm-size=2g --ulimit memlock=-1 --ulimit stack=67108864    --workdir /workspace/    -v "$(pwd):/workspace/"   nvcr.io/nvidia/tensorflow:21.10-tf2-py3
