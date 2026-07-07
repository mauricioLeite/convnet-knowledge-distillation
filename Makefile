SERVICE   = kd_lab
CONTAINER = kd_lab
IMAGE     = kd_lab:latest
JUPYTER   = http://localhost:8888

.PHONY: help build up down restart logs shell gpu cuda-check status open clean rebuild purge

help:
	@echo ""
	@echo "  Knowledge Distillation — Docker Environment"
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
	docker compose build

up:
	docker compose up -d
	@echo "JupyterLab running at $(JUPYTER)"

down:
	docker compose down

restart:
	docker compose restart $(SERVICE)

logs:
	docker compose logs -f $(SERVICE)

status:
	docker compose ps

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
	docker compose down -v --remove-orphans

rebuild:
	docker compose down
	docker compose build --no-cache
	docker compose up -d
	@echo "Rebuilt and running at $(JUPYTER)"

purge: clean
	docker rmi $(IMAGE) || true
