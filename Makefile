SHELL=sh
MAIN=build-dev

BUILD=build

CONTAINER_IMAGE=registry.gitlab.com/flaxandteal/onyx/ff_fasttext_poc:build-481688189
IMAGE_LATEST_TAG=registry.gitlab.com/flaxandteal/onyx/ff_fasttext_poc:latest
IMAGE_SHA_TAG=registry.gitlab.com/flaxandteal/onyx/ff_fasttext_poc:0321b497

GREEN  := $(shell tput -Txterm setaf 2)
YELLOW := $(shell tput -Txterm setaf 3)
WHITE  := $(shell tput -Txterm setaf 7)
CYAN   := $(shell tput -Txterm setaf 6)
RESET  := $(shell tput -Txterm sgr0)

.PHONY: all
all: build-dev

.PHONY: wheels
wheels:
	@mkdir -p $(BUILD)/wheels
	docker build -t ff_fasttext_build -f Dockerfile.wheels .
	docker run --rm --entrypoint maturin -v $(shell pwd)/$(BUILD)/wheels:/app/build/target/wheels ff_fasttext_build build

.PHONY: build
build: Dockerfile
	docker build -t ${CONTAINER_IMAGE} -t ${IMAGE_LATEST_TAG} -t ${IMAGE_SHA_TAG} .

.PHONY: build-dev
build-dev: Dockerfile
	docker-compose build

Dockerfile:
	m4 Dockerfile.in > Dockerfile

test_data/wiki.en.fifu:
	curl -o test_data/wiki.en.fifu http://www.sfs.uni-tuebingen.de/a3-public-data/finalfusion-fasttext/wiki/wiki.en.fifu

test_data/cc.cy.300.fifu:
	curl -o test_data/cc.cy.300.vec.gz https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.cy.300.vec.gz
	gunzip test_data/cc.cy.300.vec.gz
	@$(MAKE) model INPUT_VEC=test_data/cc.cy.300.vec OUTPUT_FIFU=test_data/cc.cy.300.fifu

cache/cache-cy.json:
	python translate_cache.py

.PHONY: help
help: ## Show this help.
	@echo ''
	@echo 'Usage:'
	@echo '  ${YELLOW}make${RESET} ${GREEN}<target>${RESET}'
	@echo ''
	@echo 'Targets:'
	@awk 'BEGIN {FS = ":.*?## "} { \
		if (/^[a-zA-Z_-]+:.*?##.*$$/) {printf "    ${YELLOW}%-20s${GREEN}%s${RESET}\n", $$1, $$2} \
		else if (/^## .*$$/) {printf "  ${CYAN}%s${RESET}\n", substr($$1,4)} \
		}' $(MAKEFILE_LIST)
