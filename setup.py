"""
Setup for Vectoria.

"""

from sys import platform

from setuptools import setup, find_packages
from setuptools.extension import Extension

# Package details
setup(
    name='vectoria',
    version='0.0.4',
    author='Will Ballard',
    author_email='wballard@mailframe.net',
    url='https://github.com/wballard/vectoria',
    description='Vectoria A Word Vector Encoder, used to turn word strings into dense numerical embeddings for machine learning models.',
    license='BSD 3-Clause License',
    packages=find_packages(),
    install_requires=[
        'requests>=2.13.0'
        'keras>=2.0.8',
        'tensorflow>=1.3.0',
        'numpy>=1.13.1',
        'tqdm',
        'mmh3',
        'scikit-learn>=0.18.1'
    ],
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ]
)
