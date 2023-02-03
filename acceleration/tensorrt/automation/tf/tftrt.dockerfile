ARG BASE_IMAGE=$BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /app

RUN apt-get update
RUN apt-get install -y git
RUN pip3 install --upgrade pip

RUN git clone https://github.com/lmitechnologies/LMI_AI_Solutions.git

# opencv-python-headless==4.4.0.46
RUN pip3 install opencv-contrib-python==4.5.5.64 matplotlib

# https://stackoverflow.com/questions/55313610/importerror-libgl-so-1-cannot-open-shared-object-file-no-such-file-or-directo
RUN apt-get install ffmpeg libsm6 libxext6 -y

RUN cd LMI_AI_Solutions && git fetch && git checkout wen_dev && git pull
