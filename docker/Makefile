help:
	@cat Makefile

DATA?="${HOME}/Data"
GPU?=0
DOCKER_FILE=Dockerfile
DOCKER=GPU=$(GPU) nvidia-docker
BACKEND=tensorflow
TEST=tests/
SRC=$(shell dirname `pwd`)

build:
	docker build -t bidu --build-arg python_version=3.5 -f $(DOCKER_FILE) .

bash: build
	$(DOCKER) run -it -v $(SRC):/src -v $(DATA):/data --env bidu_BACKEND=$(BACKEND) bidu bash

ipython: build
	$(DOCKER) run -it -v $(SRC):/src -v $(DATA):/data --env bidu_BACKEND=$(BACKEND) bidu ipython

notebook: build
	$(DOCKER) run -it -v $(SRC):/src -v $(DATA):/data --net=host --env bidu_BACKEND=$(BACKEND) bidu

test: build
	$(DOCKER) run -it -v $(SRC):/src -v $(DATA):/data --env bidu_BACKEND=$(BACKEND) bidu py.test $(TEST)

