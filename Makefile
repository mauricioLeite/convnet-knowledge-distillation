# MO434 — Knowledge Distillation Lab

SERVICE   = kd-lab
CONTAINER = mo434-kd
IMAGE     = mo434-kd:latest
JUPYTER   = http://localhost:8888
COMPOSE   = docker compose -f .devcontainer/docker-compose.yml

.PHONY: help build up down restart logs shell gpu cuda-check status open clean rebuild purge

help:
	@echo ""
	@echo "  MO434 Knowledge Distillation — Docker Environment"
	@echo ""
	@echo "  make build       Build the Docker image"
	@echo "  make up          Start JupyterLab (detached)"
	@echo "  make down        Stop the container"
	@echo "  make restart     Restart the container"
	@echo "  make logs        Follow container logs"
	@echo "  make shell       Open bash inside the container"
	@echo "  make gpu         Show GPU status (nvidia-smi)"
	@echo "  make cuda-check  Verify PyTorch sees the GPU"
	@echo "  make status      Show running containers"
	@echo "  make open        Print JupyterLab URL"
	@echo "  make clean       Remove container + volumes"
	@echo "  make rebuild     Full rebuild from scratch (no cache)"
	@echo "  make purge       Clean + remove the image"
	@echo ""

# Core
build:
	$(COMPOSE) build

up:
	$(COMPOSE) up -d
	@echo "JupyterLab running at $(JUPYTER)"

down:
	$(COMPOSE) down

restart:
	$(COMPOSE) restart $(SERVICE)

logs:
	$(COMPOSE) logs -f $(SERVICE)

status:
	$(COMPOSE) ps

sh:
	docker exec -it $(CONTAINER) bash

# GPU
gpu:
	docker exec -it $(CONTAINER) nvidia-smi

cuda-check:
	docker exec -it $(CONTAINER) python -c \
		"import torch; print('PyTorch:', torch.__version__); \
		 print('CUDA available:', torch.cuda.is_available()); \
		 print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

open:
	@echo "JupyterLab → $(JUPYTER)"

clean:
	$(COMPOSE) down -v --remove-orphans

rebuild:
	$(COMPOSE) down
	$(COMPOSE) build --no-cache
	$(COMPOSE) up -d
	@echo "Rebuilt and running at $(JUPYTER)"

purge: clean
	docker rmi $(IMAGE) || true
