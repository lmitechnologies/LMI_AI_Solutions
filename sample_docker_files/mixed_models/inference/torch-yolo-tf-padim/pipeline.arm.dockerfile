FROM --platform=linux/arm64/v8 nvcr.io/nvidia/l4t-ml:r35.1.0-py3

ARG PACKAGE_VER
ARG PYPI_SERVER

ARG DEBIAN_FRONTEND=noninteractive

ENV LANG C.UTF-8
ENV LD_PRELOAD /lib/aarch64-linux-gnu/libGLdispatch.so.0:/lib/aarch64-linux-gnu/libgomp.so.1

WORKDIR /home/gadget/workspace

RUN apt-get update && apt-get install git -y
RUN pip install --upgrade pip

RUN pip3 install --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v50 tensorflow==2.9.1+nv22.6

RUN git clone https://github.com/lmitechnologies/LMI_AI_Solutions.git && cd LMI_AI_Solutions && git submodule update --init object_detectors/yolov5

# RUN pip install -r LMI_AI_Solutions/object_detectors/yolov5/requirements.txt
# alternative minimum dependencies.
RUN pip install tqdm>=4.64.0 seaborn>=0.11.0

COPY ./tensorflow_addons-0.20.0.dev0-cp38-cp38-linux_aarch64.whl ./
RUN pip3 install ./tensorflow_addons-0.20.0.dev0-cp38-cp38-linux_aarch64.whl

