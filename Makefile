
BUILD_TAG=latest
# name for docker image
DOCKER_IMAGE=odahu/sagemaker-mlflow-container
# tag for docker image
TAG=
# Example of DOCKER_REGISTRY: nexus.domain.com:443/
DOCKER_REGISTRY=

# pytest required parameters for integration tests
TEST_SM_ROLE=
TEST_IMAGE=

-include .env

.EXPORT_ALL_VARIABLES:

check-tag:
	@if [ "${TAG}" == "" ]; then \
	    echo "TAG is not defined, please define the TAG variable" ; exit 1 ;\
	fi
	@if [ "${DOCKER_REGISTRY}" == "" ]; then \
	    echo "DOCKER_REGISTRY is not defined, please define the DOCKER_REGISTRY variable" ; exit 1 ;\
	fi

# install project in setuptools editable mode
install_editable:
	pip install -e .

# install project in setuptools editable mode with extra deps for tests
install_editable_test:
	pip install -e ".[test]"

# run pytest unit tests
tests_unit:
	pytest -s tests/unit

# run pytest integration tests
tests_integration:
	pytest -s tests/integration --sagemaker-role ${TEST_SM_ROLE} --image ${TEST_IMAGE}

lint:
	flake8 src/sagemaker_mlflow_container
	flake8 tests --exclude=resources

# build docker image
docker-build:
	docker build -t ${DOCKER_IMAGE}:${BUILD_TAG} . -f containers/app/Dockerfile

# create repository for DOCKER_IMAGE if doesn't exists
ecr-prepare:
	scripts/ecr_prepare.sh ${DOCKER_IMAGE}

# push docker image
docker-push: check-tag
	docker tag ${DOCKER_IMAGE}:${BUILD_TAG} ${DOCKER_REGISTRY}/${DOCKER_IMAGE}:${TAG}
	docker push ${DOCKER_REGISTRY}/${DOCKER_IMAGE}:${TAG}

clean:
	-rm -r dist
	-rm -r *.egg-info
	-rm -r mlruns