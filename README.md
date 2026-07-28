# Small Students, Big Lessons: Distilling Pretrained Backbones into Lightweight ConvNets

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-supported-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository contains the code for a knowledge distillation study from
pretrained image classifiers into lightweight convolutional student models.

The official report is available on [Zenodo](https://zenodo.org/records/21521918).

The experiments compare:

- Teacher backbones: VGG16-BN, ResNet-50, and ConvNeXt-Base.
- Datasets: Oxford-IIIT Pet, Flowers-102, and Tiny ImageNet.
- Distillation targets: pre-GAP feature maps and post-GAP pooled vectors.
- Student architectures with different depth, residual structure, parameter
  count, and compute cost.
- Loss configurations combining MSE feature imitation, cross-entropy, softened
  logit KD, and neighborhood relational KD.

Generated outputs, checkpoints, logs, and downloaded datasets are intentionally
not part of the minimal source tree. They can be regenerated with the commands
below.

## Repository Layout

```text
.
├── data/
│   └── download_datasets.py
├── src/
│   ├── common/
│   ├── phase1/
│   ├── phase2/
│   └── phase3/
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── requirements.txt
└── README.md
```

Main phases:

- `src/phase1/`: teacher model setup, teacher training, and teacher feature
  extraction utilities.
- `src/phase2/`: student definitions, distillation losses, student generation,
  and student training.
- `src/phase3/`: evaluation, result aggregation, plots, ablation analysis, and
  student-teacher activation rendering.

## Requirements

The recommended environment is Docker with GPU support.

Host requirements:

- Docker and Docker Compose.
- NVIDIA GPU drivers.
- NVIDIA Container Toolkit.

The Docker image is based on CUDA 12.8 and installs Python 3.11, PyTorch,
TorchVision, JupyterLab, and the Python packages listed in `requirements.txt`.

## Setup

Build the environment:

```bash
make build
```

Start JupyterLab and the project container:

```bash
make up
```

The default JupyterLab URL is:

```text
http://localhost:8888
```

JupyterLab is configured without a token or password. Use it only on a trusted
local machine, or add authentication before exposing port 8888 on a network.

Check that CUDA is visible inside the container:

```bash
make cuda-check
```

## Datasets

Download the supported datasets:

```bash
docker exec -w /workspace kd_lab python data/download_datasets.py
```

The script downloads/prepares:

- Tiny ImageNet.
- Flowers-102.
- Oxford-IIIT Pet.

Datasets are stored under `data/` and are not meant to be committed.

## Reproducing The Experiments

Train teacher models:

```bash
docker exec -w /workspace kd_lab python src/phase1/train_teachers.py
```

Generate student configurations:

```bash
docker exec -w /workspace kd_lab python src/phase2/gen_students.py \
  --datasets oxford-pets flowers-102 tiny-imagenet-200
```

Train students:

```bash
docker exec -w /workspace kd_lab python src/phase2/train_students.py \
  --teachers resnet convnext vgg \
  --datasets oxford-pets flowers-102 tiny-imagenet-200
```

The commands must be run in the order shown: student training requires both
the generated student configuration files and the teacher checkpoints; student
evaluation requires the resulting student weights and `summary.json` files.

Evaluate students:

```bash
docker exec -w /workspace kd_lab python src/phase3/test_students.py \
  --datasets oxford-pets flowers-102 tiny-imagenet-200
```

Run the loss ablation script:

```bash
docker exec -w /workspace kd_lab bash src/phase2/run_ablation.sh
```

This runs six loss configurations for the `arch6_6conv_res` pre-GAP students,
across all three teachers and datasets. It is substantially more expensive than
a single training run.

Regenerate plots and analysis tables:

```bash
docker exec -w /workspace kd_lab python src/phase3/plot_results.py --no-title
docker exec -w /workspace kd_lab python src/phase3/analyze_ablation.py --no-title
```

Render student-teacher activation visualizations:

```bash
docker exec -w /workspace kd_lab python src/phase3/render_student_activations.py \
  --datasets oxford-pets flowers-102 tiny-imagenet-200 \
  --ids "arch6_6conv_res__*_pre_gap" \
  --image-indices 0 50 100 \
  --num-workers 0
```

## Convenience Commands

The `Makefile` includes shortcuts for common tasks:

```bash
make build              # Build Docker image
make up                 # Start JupyterLab/container
make down               # Stop container
make logs               # Follow container logs
make sh                 # Open a shell in the container
make gpu                # Show GPU status
make cuda-check         # Verify PyTorch CUDA access
make status             # Show container status
make open               # Print the JupyterLab URL
```

## Outputs

Generated artifacts are written under `outputs/`, including:

- `outputs/checkpoints/`
- `outputs/students/`
- `outputs/tables/`
- `outputs/figures/`

These artifacts can be large and are excluded from version control.

## License

This project is released under the MIT License.
