#
#    Copyright 2020 EPAM Systems
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
#
import os

# Conda environment where MLProject conda dependencies will be installed
# This environment will be used to run mlflow training
# conda run -n $CONDA_TRAINING_ENV mlflow run <MLProject dir>
CONDA_TRAINING_ENV = os.environ.get('CONDA_TRAINING_ENV', 'training')

# Key in JSON document returned by `codna info` command which indicates
# the path to conda environment location
CONDA_INFO_ENV_PATH_KEY = 'active_prefix'


# Prefix of SageMaker Estimator hyperparameters that relate to tuning of mlflow running
# but not to hyperparameters of training script itself
MLFLOW_RUN_PARAMS_PREFIX = 'sagemaker_mlflow_run_'


# Parameter of `mlflow run ...` that is used to specify MLFlow run-id
MLFLOW_RUN_ID_PARAM = 'run-id'

# all artifacts saved during MLFlow training run will be saved into this subdir
SAGEMAKER_MODEL_SUBDIR = 'mlflow_run_artifacts'
