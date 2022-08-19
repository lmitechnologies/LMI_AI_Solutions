# fringeai-tftrt Baseline Workflow
Conversion from Tensorflow to TensorRT.
Uses an Nvidia docker container that will run on an x86 host system

Sample files for:
   - pretrained resnet50 classifier
   - generic saved model with input image directory

# Installation

Clone repo:
```
git clone git@github.com:lmitechnologies/fringeai-tftrt.git
```
Navigate to baseline_workflow:
```
cd baseline_workflow
```
Get data files from GCP:
```
gsutil -m cp -r gs://internal_projects/tensorrt_files/trained-inference-models .
gsutil -m cp -r gs://internal_projects/tensorrt_files/data .
```

# Launch Docker Container CLI

Launch Docker: 
```
launch_docker.bash
```
Install dependencies at container prompt:
```
install_dependencies.bash
```
Run classifier:
```
python3 resnet50_mnist.py --design_baseline --generate_trt --benchmark_baseline --benchmark_trt
```
Run object detector:
```
python3 savedmodel_withbuild.py --generate_trt --benchmark_baseline --benchmark_trt
```

# Pre-Requisites:
## 1) Install Docker

### a) Update and Install
```bash
sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io
```

### b) Add $USER to Docker Group
```bash
sudo usermod -aG docker $USER
```

### c) Logout and Log Back In


### d) Test Docker
```bash
sudo docker run hello-world
```

you should see something like this:

```
Unable to find image 'hello-world:latest' locally
latest: Pulling from library/hello-world
2db29710123e: Pull complete 
Digest: sha256:37a0b92b08d4919615c3ee023f7ddb068d12b8387475d64c622ac30f45c29c51
Status: Downloaded newer image for hello-world:latest

Hello from Docker!
This message shows that your installation appears to be working correctly.

To generate this message, Docker took the following steps:
 1. The Docker client contacted the Docker daemon.
 2. The Docker daemon pulled the "hello-world" image from the Docker Hub.
    (amd64)
 3. The Docker daemon created a new container from that image which runs the
    executable that produces the output you are currently reading.
 4. The Docker daemon streamed that output to the Docker client, which sent it
    to your terminal.

To try something more ambitious, you can run an Ubuntu container with:
 $ docker run -it ubuntu bash

Share images, automate workflows, and more with a free Docker ID:
 https://hub.docker.com/

For more examples and ideas, visit:
 https://docs.docker.com/get-started/
```

## 2) Install NVIDIA Container Toolkit
### a) Setup stable repository and GPG Key
```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
   && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
```
### b) Update and Install
```bash
sudo apt-get update \
    && sudo apt-get install -y nvidia-docker2
```
### c) Restart Docker
```bash
sudo sysemctl restart docker
```
### d) Test NVIDIA Container Toolkit
```bash
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```

You should see output like this:

```
Mon Nov  1 21:46:54 2021       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 460.91.03    Driver Version: 460.91.03    CUDA Version: 11.2     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  GeForce RTX 208...  Off  | 00000000:03:00.0 Off |                  N/A |
| 22%   29C    P8     5W / 260W |    199MiB / 11018MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
+-----------------------------------------------------------------------------+



