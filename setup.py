from __future__ import absolute_import

import os

from setuptools import find_packages, setup


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name='sagemaker_mlflow_container',
    version=read('VERSION'),
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    description='Open source library for creating MLFlow containers to run on Amazon SageMaker.',
    author='Vlad Tokarev, Vitalik Solodilov',
    author_email='vlad.tokarev.94@gmail.com, mcdkr@yandex.ru',
    license='Apache v2',
    python_requires='>=3.6',

    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python",
        'Programming Language :: Python :: 3.6'
    ],

    install_requires=['sagemaker-containers>=2.8.6', 'PyYAML>=3.1.2', 'mlflow>=1.7', 'urllib3'],
    extras_require={
        'test': ['pytest', 'sagemaker>=1.55.2', 'flake8']
    },
)
