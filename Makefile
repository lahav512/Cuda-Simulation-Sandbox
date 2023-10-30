.PHONY: build run push save upload load images

dir_name := $(shell basename $(CURDIR) | tr 'A-Z' 'a-z')
ssh_url := "gpu-sandbox-git"

all:
	@echo ${dir_name}

build:
	docker buildx build --pull=false --load --platform linux/amd64 -t ${dir_name} .
	# $(MAKE) load
	# $(MAKE) images


run:
	docker run -it ${dir_name}

push:
	$(MAKE) save
	$(MAKE) upload

save:
	docker save "${dir_name}:latest" | gzip > "${dir_name}-image.tar.gz"

upload:
	# scp ${dir_name}-image.tar.gz ${ssh_url}:~
	rsync -avz --progress ${dir_name}-image.tar.gz ${ssh_url}:~

load:
	docker load -i ${dir_name}.tar

images:
	docker images ${dir_name}

