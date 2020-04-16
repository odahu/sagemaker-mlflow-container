import os
from contextlib import contextmanager
from urllib.parse import urlparse

import boto3
from sagemaker.estimator import Estimator, DIR_PARAM_NAME
from sagemaker.utils import create_tar_file


@contextmanager
def change_dir(new_dir):
    old_dir = os.getcwd()
    os.chdir(new_dir)
    yield
    os.chdir(old_dir)


def test_run(resources_folder, sagemaker_role, image, tmpdir, sm_session, key_prefix):

    with change_dir(os.path.join(resources_folder, 'ml/code')):
        tar_file = create_tar_file(
            os.listdir(), target=tmpdir / 'code.tar.gz'
        )

    s3_uri = sm_session.upload_data(
        path=str(tar_file),
        key_prefix=key_prefix
    )

    estimator = Estimator(image_name=image,
                          hyperparameters={
                              DIR_PARAM_NAME: s3_uri,
                              'alpha': '1.0',
                              'sagemaker_mlflow_experiment_id': '2.0',
                          },
                          role=sagemaker_role,
                          train_instance_count=1,
                          train_instance_type='ml.m4.xlarge')

    estimator.fit()

    _assert_s3_file_exists(sm_session.boto_region_name, estimator.model_data)


def _assert_s3_file_exists(region, s3_url):
    parsed_url = urlparse(s3_url)
    s3 = boto3.resource('s3', region_name=region)
    s3.Object(parsed_url.netloc, parsed_url.path.lstrip('/')).load()
