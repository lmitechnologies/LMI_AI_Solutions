
# pytorch:22.07-py3 - (dpkg -l | grep TensorRT: 8.4.1-1+cuda11.6), 
# but pip install tensorflow updates it to 8.5.1-1+cuda11.8 which leads to inconsistency
# so use nvidia/tensorflow instead.
FROM --platform=linux/amd64 nvcr.io/nvidia/tensorflow:22.07-tf2-py3

ARG DEBIAN_FRONTEND=noninteractive

WORKDIR /home/gadget/workspace

RUN apt-get update
RUN pip install --upgrade pip

RUN git clone https://github.com/lmitechnologies/LMI_AI_Solutions.git && cd LMI_AI_Solutions && git submodule update --init object_detectors/yolov5
RUN pip install -r LMI_AI_Solutions/object_detectors/yolov5/requirements.txt

# fix - ImportError: libGL.so.1: cannot open shared object file
RUN apt-get install libgl1 -y
