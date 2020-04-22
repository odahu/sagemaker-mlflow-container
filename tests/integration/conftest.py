import os
import uuid
from os import path

import boto3
import pytest
import sagemaker

RESOURCES = 'resources'


def pytest_addoption(parser):
    parser.addoption('--sagemaker-role', required=True,
                     help='SageMaker role that will be assumed by sagemaker sdk Estimator class '
                          'to interact with SageMaker API. This role is required even to launch '
                          'local training')
    parser.addoption('--image', required=True,
                     help='container image for train running')


@pytest.fixture(scope='session')
def resources_folder():
    # we need to have ability to pass resources folder from host (while testing in docker)
    default_resources = path.join(path.dirname(__file__), RESOURCES)
    resources = os.environ.get('TEST_RESOURCES', default_resources)
    return resources


@pytest.fixture
def sagemaker_role(request):
    return request.config.getoption('--sagemaker-role')


@pytest.fixture
def image(request):
    return request.config.getoption('--image')


@pytest.fixture(scope='session')
def sm_session():
    sm_session = sagemaker.Session()
    yield sm_session
    # clean default bucket
    s3 = boto3.resource('s3', region_name=sm_session.boto_region_name)
    bucket = s3.Bucket(sm_session.default_bucket())
    bucket.objects.all().delete()


@pytest.fixture
def uuid_str():
    return str(uuid.uuid4()).replace('-', '')


@pytest.fixture
def key_prefix(sm_session, uuid_str):
    yield f'submitted-code-{uuid_str}'
