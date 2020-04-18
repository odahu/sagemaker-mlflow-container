
### Introduction

AWS SageMaker compatible container to run mlflow trainings.

The container is created with using [Amazon SageMaker Containers] library

### Quickstart

Requirements:

    - Python >=3.6
    - pip
    - aws cli
    - aws account

Create quickstart project dir

```bash
$ mkdir sm-mlflow-quickstart && cd sm-mlflow-quickstart
```

Clone mlflow repo to get sources of MLFlow project examples and set env var to one of examples

```bash
$ git clone git@github.com:mlflow/mlflow.git
MLFLOW_PROJECT_PATH=mlflow/examples/sklearn_elasticnet_wine
```

Install sagemaker python sdk

```bash
pip install sagemaker
```

Authorize in AWS SageMaker using AWS CLI 

Create SageMaker execution role using AWS Console and export it as env var
```bash
export SAGEMAKER_ROLE=`your-role`
```

Create `control_script.py`

```python
import os
from contextlib import contextmanager

import sagemaker
from sagemaker.estimator import Estimator, DIR_PARAM_NAME
from sagemaker.utils import create_tar_file

# fetch sagemaker role and mlflow project path
sagemaker_role = os.environ.get('SAGEMAKER_ROLE', 'SageMakerRole')
mlflow_project_path = os.environ.get('MLFLOW_PROJECT_PATH')

# sagemaker-mlflow-conainer built and pushed to your ECR image name
image = 'ecr-image-reference'

# helper function
@contextmanager
def change_dir(new_dir):
    old_dir = os.getcwd()
    os.chdir(new_dir)
    yield
    os.chdir(old_dir)

if __name__ == '__main__':
    mlflow_project_dir = f'file://{mlflow_project_path}'
    
    sm_session = sagemaker.Session()

    with change_dir(mlflow_project_path):
        tar_file = create_tar_file(
            os.listdir(), target='code.tar.gz'
        )
    
    key_prefix = 'submitted_code'
    
    # upload code from mlflow example to s3
    s3_uri = sm_session.upload_data(
        path=str(tar_file),
        key_prefix=key_prefix
    )
    
    # describe training
    estimator = Estimator(image_name=image,
                          hyperparameters={
                              DIR_PARAM_NAME: s3_uri,
                              'alpha': '1.0',
                              'sagemaker_mlflow_experiment_id': '2.0',
                          },
                          role=sagemaker_role,
                          train_instance_count=1,
                          train_instance_type='ml.m4.xlarge')
    
    # run training
    estimator.fit()
    
    # print s3 uri to MLFlow logged models
    print(estimator.model_data)
```

Run training
```bash
python control_script.py
```

s3 uri with all mlflow logged models will be printed

### Tasks

This section describes how to solve different tasks that you usually should to solve while run training

#### How to pass MLFlow entrypoint parameters?

mlflow cli support passing entrypoint parameters and other training script command line 
arguments using [`-P, --param-list` parameter](https://www.mlflow.org/docs/latest/cli.html#cmdoption-mlflow-run-p)

These parameters should be passed using `hyperparameters` parameter of [`Estimator` class](https://sagemaker.readthedocs.io/en/stable/estimators.html#sagemaker.estimator.Estimator)      


#### How to set `experiment_id`, `run_id` or other `mlflow run` parameters?

You can customize any extra parameters that are passed into `mlflow run` using 
`sagemaker_mlflow_run_*` prefixed reserved hyperparameters as described in reference section


#### How to avoid updating conda environment every training run?

Inherit from base docker image and install required packages into conda environment with
a name "training" or create another one and set $CONDA_TRAINING_ENV to new conda env name

#### How to set tracking uri

Inherit from base docker image and override $MLFLOW_TRACKING_URI environment variable

### Reference

#### Reserved `Estimator` hyperparameters

All `sagemaker_*` prefixed hyperparameters are reserved as sagemaker or framework hyperparameters

1. See list of `sagemaker_*` prefixed [Amazon SageMaker Containers] 
reserved hyperparameters on [official reference](https://github.com/aws/sagemaker-containers/blob/master/TRAINING_IN_DETAIL.rst)

2. `sagemaker_mlflow_run` (sagemaker_mlflow_run_*) prefixed hyperparameters are reserved 
by the container to pass additional parameters to `mlflow run` command


[Amazon SageMaker Containers]: https://docs.aws.amazon.com/sagemaker/latest/dg/amazon-sagemaker-containers.html
