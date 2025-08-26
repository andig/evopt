DOCKER_IMAGE := andig/evopt

default: build docker-build

build::
	go generate ./...

test::
	go run example/client.go

run::
	python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt && python3 app.py

docker: docker-build docker-run

docker-build::
	docker buildx build . --tag $(DOCKER_IMAGE) --push

docker-run::
	docker run -p 7050:7050 -it $(DOCKER_IMAGE)

fly::
	fly deploy --local-only
