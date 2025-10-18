DOCKER_IMAGE := andig/evopt

default: build docker-build

build::
	go generate ./...

install::
	uv sync

upgrade::
	uv lock --upgrade

lint::
	uv run autopep8 --in-place --recursive .
	uv run ruff check --fix

test::
	uv run pytest

run::
	uv run python -m evopt.app

run-gunicorn::
	uv run gunicorn --bind "0.0.0.0:7050" "evopt.app:app"

loadtest::
	uv run locust --host http://localhost:7050 --headless -t 30s -u 2 --only-summary
	uv run locust --host http://localhost:7050 --headless -t 30s -u 4 --only-summary

docker: docker-build docker-push docker-run

docker-build::
	docker buildx build . --tag $(DOCKER_IMAGE)

docker-run::
	docker run -p 7050:7050 -it $(DOCKER_IMAGE)

docker-push::
	docker push $(DOCKER_IMAGE)

fly::
	fly deploy --local-only
