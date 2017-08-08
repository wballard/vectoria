"""
Setup for Vectoria. This uses cython, so will compile the C++ code of
FastText into a python extension.
"""

from sys import platform

import numpy as np
from Cython.Build import cythonize

from setuptools import setup
from setuptools.extension import Extension

# Define the C++ extension
if platform == "darwin":
    EXTRA_COMPILE_ARGS = ['-O3', '-pthread', '-std=c++11', '-static', '-mmacosx-version-min=10.7']
else:
    EXTRA_COMPILE_ARGS = ['-O3', '-pthread', '-std=c++11', '-static']

EXTENSIONS = [
    Extension('*',
              sources=[
                  'vecoder/fasttext.pyx',
                  'vecoder/fasttext/args.cc',
                  'vecoder/fasttext/dictionary.cc',
                  'vecoder/fasttext/fasttext.cc',
                  'vecoder/fasttext/matrix.cc',
                  'vecoder/fasttext/model.cc',
                  'vecoder/fasttext/productquantizer.cc',
                  'vecoder/fasttext/qmatrix.cc',
                  'vecoder/fasttext/utils.cc',
                  'vecoder/fasttext/vector.cc',
              ],
              language='c++',
              extra_compile_args=EXTRA_COMPILE_ARGS)
]

# Package details
setup(
    name='vecoder',
    version='0.0.1',
    author='Will Ballard',
    author_email='wballard@mailframe.net',
    url='https://github.com/wballard/vecoder',
    description='Word Vector Encoder',
    long_description=open('README.rst', 'r').read(),
    license='BSD 3-Clause License',
    packages=['vecoder'],
    ext_modules=cythonize(EXTENSIONS),
    include_dirs=['.', np.get_include()],
    install_requires=[
        'numpy>=1',
        'future'
    ],
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: C++',
        'Programming Language :: Cython',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.2',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ]
)
