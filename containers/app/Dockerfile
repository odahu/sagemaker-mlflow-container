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

FROM continuumio/miniconda3:4.8.2
RUN apt-get install -y gcc

ARG PIP_EXTRA_INDEX_URL

# Environment for running mlflow trainings
RUN conda create -n runner python=3.6
RUN conda init bash
RUN echo "conda activate runner" >> ~/.bashrc

COPY src /opt/runner/src
COPY setup.py /opt/runner
COPY README.md /opt/runner
COPY VERSION /opt/runner

RUN pip install -e /opt/runner

# Pre-create isolated conda environment for running mlflow training
RUN conda create -n training python --no-default-packages

ENV SAGEMAKER_TRAINING_MODULE=sagemaker_mlflow_container.training:main

CMD [ "/bin/bash" ]
