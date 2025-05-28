# Object Detectors

## 1. Pre Labeling with Segment Anything Model 2

### 1.1 Clone the sam lmi repo
```bash
git clone https://github.com/lmitechnologies/sam.git
```

### 1.2 sam lmi repo strcture
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

This document focus on the `v2` folder including the scripts related to the sam2 model. It's content is listed below:

`configs`: the folder contains the sam2 configuration files.  
`weights`: the folder contains the pretrained sam2 weights files.  
`automatic_mask_generator.py`: the script to automatically generate masks for all the found objects in images.  
`prompt_with_similarity2.py`: the script to interactive select points for objects in the first image and automatically find similar objects in the remaining images.  
`prompt.py`: the script to manually select points of target objects for all the images.  

Note: `prompt_with_similarity2.py` and `prompt.py` are tested inside docker in a **windows** host.

### 1.3 modify the docker-compose.yaml file

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

