# MO434 — Knowledge Distillation Lab
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=America/Sao_Paulo \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common ca-certificates gnupg && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3.11-venv python3.11-distutils \
        git wget curl unzip build-essential \
        libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python  python  /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 && \
    python -m pip install --upgrade pip setuptools wheel

WORKDIR /workspace
COPY requirements.txt /tmp/requirements.txt

RUN pip install --index-url https://download.pytorch.org/whl/cu128 \
        torch torchvision torchaudio && \
    pip install -r /tmp/requirements.txt

RUN mkdir -p /root/.jupyter && \
    printf "c.ServerApp.token = ''\nc.ServerApp.password = ''\nc.ServerApp.allow_root = True\nc.ServerApp.ip = '0.0.0.0'\nc.ServerApp.open_browser = False\n" \
        > /root/.jupyter/jupyter_server_config.py

EXPOSE 8888
CMD ["bash"]
