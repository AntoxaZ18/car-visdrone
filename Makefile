.PHONY: docker_build docker_run venv_init help

DOCKER_IMAGE = cuda11
JUPYTER_PORT = 8888

help:
	@echo "Usage:"
	@echo "  make <target>"
	@echo ""
	@echo " Targets:"
	@echo "  help            - Show this help message"
	@echo "  docker_build    - Build docker image"
	@echo "  docker_run   	 - run docker container"
	@echo "  docker_clean    - delete image"
	@echo "  venv_init       - init venv with poetry"

docker_build:
	@echo "Docker start to build..."
	docker buildx build . -t $(DOCKER_IMAGE) --progress plain

docker_run:
	@echo "Run Docker container... It will be deleted automatically after stopping"
	docker run -p $(JUPYTER_PORT):$(JUPYTER_PORT) --rm --gpus all -v .:/app $(DOCKER_IMAGE) 

docker_clean:
	@echo "Image delete..."
	docker rmi $(DOCKER_IMAGE):latest -f

venv_init:
	@echo "init venv"
	poetry config virtualenvs.in-project true
	poetry install
	poetry env activate

clean:
	@echo "Очистка проекта..."
	rm -rf .venv/
	rm -rf ./dataset
	rm -rf __pycache__
	rm -rf ./projects
	rm -rf ./runs
	rm -rf ./onnx_models
