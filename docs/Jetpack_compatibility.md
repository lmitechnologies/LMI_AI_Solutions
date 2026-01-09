# AI Models Compatibility with NVIDIA Jetpack Versions

This document provides compatibility information for AI models and frameworks on NVIDIA Jetson devices across different Jetpack versions.

## Overview

This guide covers compatibility for building AI solution pipelines using:
- Detectron2
- Ultralytics YOLO
- Anomalib: v1.1.1

Across model formats:
- PT
- TorchScript
- TensorRT

## Jetpack Versions Covered

- **Jetpack 4.5** - Based on Ubuntu 18.04, CUDA 10.2
- **Jetpack 5.1** - Based on Ubuntu 20.04, CUDA 11.4

---

## Compatibility Tables

### Detectron2 Compatibility

| Model Format | Jetpack 4.5 | Jetpack 5.1 |
|--------------|-------------|-------------|
| PT (PyTorch) | ❌ Does not work | ❔ Not tested |
| TorchScript | ❌ Does not work | ✅ Works |
| TensorRT | ❌ Does not work | ❔ Not tested |

---

### Ultralytics YOLO Compatibility

| Model Format | Jetpack 4.5 | Jetpack 5.1 |
|--------------|-------------|-------------|
| PT (PyTorch) | ✅ Works | ✅ Works |
| TorchScript | ✅ Works | ✅ Works |
| TensorRT | ✅ Works | ✅ Works |

---

### Anomalib v1 Compatibility

| Model Format | Jetpack 4.5 | Jetpack 5.1 |
|--------------|-------------|-------------|
| PT (PyTorch) | ❌ Does not work | ❌ Does not work |
| TorchScript | ❌ Does not work | ✅ Works |
| TensorRT | ❌ Does not work | ✅ Works |

---

*Last Updated: January 9 2026*