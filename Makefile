.PHONY: build run push


build:
	docker buildx build --load --platform linux/amd64 -t cuda-sandbox .
	# docker buildx create --use
	# docker buildx build --platform linux/amd64,linux/arm64 -t cuda-sandbox:latest .


run:
	docker run -it cuda-sandbox

push:
	docker save cuda-sandbox:latest | gzip > cuda-sandbox-image.tar.gz
	scp cuda-sandbox-image.tar.gz simulation-sandbox-ec2-git:~

