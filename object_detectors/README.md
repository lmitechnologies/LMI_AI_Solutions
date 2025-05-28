# Object Detectors

## 1. Pre Labeling with Segment Anything Model 2

### 1.1 Clone the Repo
```bash
git clone https://github.com/lmitechnologies/sam.git
```

### 1.2 Repo Structure
```
├── data
├── v1 (deprecated)
├── v2
    ├── configs
    ├── weights
    ├── automatic_mask_generator.py
    ├── docker-compose.yaml
    ├── dockerfile
    ├── prompt_with_similarity2.py
    ├── prompt.py
├── my_utils.py
```

This document focuses on the `v2` folder, which contains scripts related to the sam2 model. The contents of this folder are as follows:

`configs`: this folder contains the sam2 configuration files.  
`weights`: this folder contains the pretrained sam2 weights files.  
`automatic_mask_generator.py`: this script automatically generates masks for all detected objects within images.  
`prompt_with_similarity2.py`: this script allows for interactively point selection for objects in the first image and automatically find similar objects in subsequent images.  
`prompt.py`: this script allows manually point selection for target objects across all images.  


### 1.3 Auto Mask Generation

Modify `v2/docker-compose.yaml` file:
- Modify the path to input folder
- Replace the `class-0` with the real class name
- Modify the path to output folder
- (optional) modify configs and weights files. The model config and weights files are default to `sam2.1_hiera_base_plus`. Check more at https://github.com/facebookresearch/sam2.

```yaml
services:
  sam2-pre-label:
    container_name: sam2-pre-label
    build:
      context: .
      dockerfile: ./dockerfile
    ipc: host
    runtime: nvidia # ensure that Nvidia Container Toolkit is installed
    environment:
      - DISPLAY=192.168.1.8:0.0   # replace with your host's IP address for prompting ONLY
    volumes:
      - ../data/samples:/app/data
      - ./configs:/app/configs
      - ./weights:/app/weights
      - ./outputs/samples:/app/outputs
      - ./automatic_mask_generator.py:/app/run.py
      - ../my_utils.py:/app/my_utils.py
    command: >
      bash -c "python /app/run.py -i /app/data -o /app/outputs -c class-0"

```

### 1.4 Prompting
These instructions explain how to set up your environment to enable GUI-based prompting, likely from within a Docker container that requires an X server for display.

#### 1.4.1 Running on a Windows host
To display graphical applications from the Docker container on your Windows host, you'll need to install and configure an X server.

1. Install VcXsrv (X server for Windows) from https://vcxsrv.com/.
2. Launch XLaunch from your Start Menu.
3. Go through the configuration wizard with these settings:
    - Display settings: Choose "Multiple windows".
    - Display number: Leave it as 0.
    - Client startup: Choose "Start no client".
    - Extra settings: Crucially, check the box for "Disable access control". This allows the Docker container to connect to it.
4. Find Your Host's IP Address.
5. Replace the IP in the docker compose file.

#### 1.4.2 Running on a Linux host
Ensure your Linux host's X server is configured to accept connections from the Docker container.

1. Allow Connections to Your X Server by running this command in your terminal:
    ```bash
    xhost +
    ```
2. Modify the docker-compose file:
    - Modify the `DISPLAY` environmental variable: 
      ```yaml
      environment:
        - DISPLAY=${DISPLAY}
      ```
    - Add a volume for X11 communication:
      ```yaml
      volumes:
        - /tmp/.X11-unix:/tmp/.X11-unix
      ```
