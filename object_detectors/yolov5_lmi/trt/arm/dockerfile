# jetpack 5.0.2
FROM --platform=linux/arm64/v8 nvcr.io/nvidia/l4t-ml:r35.1.0-py3

ARG DEBIAN_FRONTEND=noninteractive
ENV LANG C.UTF-8

WORKDIR /repos
RUN pip3 install --upgrade pip
RUN git clone https://github.com/lmitechnologies/LMI_AI_Solutions.git && cd LMI_AI_Solutions && git submodule update --init object_detectors/submodules/yolov5

RUN pip3 install opencv-python==4.6.0.66 --user
RUN pip3 install -r /repos/LMI_AI_Solutions/object_detectors/submodules/yolov5/requirements.txt
