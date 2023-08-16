# last version using ubuntu 20.04, require CUDA 12.1 
FROM nvcr.io/nvidia/pytorch:23.04-py3
ARG DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y
RUN pip install --upgrade pip setuptools wheel
RUN pip install --user  opencv-python
RUN pip install ultralytics -U

# clone AIS
WORKDIR /repos
RUN git clone https://github.com/lmitechnologies/LMI_AI_Solutions.git
