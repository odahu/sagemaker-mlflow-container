FROM 505502776526.dkr.ecr.eu-central-1.amazonaws.com/odahu/sagemaker-mlflow-container

COPY conda.yaml /opt/custom-deps/conda.yaml
RUN conda env update -n training -f /opt/custom-deps/conda.yaml