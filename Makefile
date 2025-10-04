DOCKER_IMAGE := andig/evopt

default: build docker-build

build::
	go generate ./...

install::
	uv sync

upgrade::
	uv lock --upgrade

lint::
	uv run ruff format
	uv run ruff check --fix

test::
	uv run pytest

run::
	uv run python -m evopt.app

docker: docker-build docker-push docker-run

docker-build::
	docker buildx build . --tag $(DOCKER_IMAGE)

docker-run::
	docker run -p 7050:7050 -it $(DOCKER_IMAGE)

docker-push::
	docker push $(DOCKER_IMAGE)

fly::
	fly deploy --local-only
