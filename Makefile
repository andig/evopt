DOCKER_IMAGE := andig/evopt

default: build docker-build

build::
	go generate ./...

test::
	go run example/client.go

install::
	python3 -m venv .venv && source .venv/bin/activate && pip install --upgrade -r requirements.txt

upgrade::
	python3 -m venv .venv && source .venv/bin/activate && pip install --upgrade $$(pip list --outdated | awk 'NR>2 {print $$1}')

run::
	source .venv/bin/activate && python3 app.py

docker: docker-build docker-run

docker-build::
	docker buildx build . --tag $(DOCKER_IMAGE) --push

docker-run::
	docker run -p 7050:7050 -it $(DOCKER_IMAGE)

fly::
	fly deploy --local-only
