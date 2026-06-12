# MO434 — Knowledge Distillation Lab
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=America/Sao_Paulo \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System packages + Python 3.11 (pyproject now allows >=3.11).
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common ca-certificates gnupg && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3.11-venv \
        git wget curl unzip build-essential \
        libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# uv — same project/lockfile workflow as the host (`uv run ...`).
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# Install deps into a venv OUTSIDE /workspace so the bind-mounted host .venv
# (Windows) never shadows it. Torch (cu126, from the lock) runs fine on the
# CUDA 12.8 base. /opt/venv/bin on PATH -> `python`/`jupyter` resolve to it.
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    PATH=/opt/venv/bin:$PATH

WORKDIR /workspace
# Only the manifests are needed at build time; source is bind-mounted at runtime.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --python /usr/bin/python3.11 && \
    uv pip install --python /opt/venv jupyterlab

# Tokenless local JupyterLab.
RUN mkdir -p /root/.jupyter && \
    printf "c.ServerApp.token = ''\nc.ServerApp.password = ''\nc.ServerApp.allow_root = True\nc.ServerApp.ip = '0.0.0.0'\nc.ServerApp.open_browser = False\n" \
        > /root/.jupyter/jupyter_server_config.py

EXPOSE 8888
CMD ["bash"]
