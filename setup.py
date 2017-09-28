"""
Setup for Vectoria. This uses cython, so will compile the C++ code of
FastText into a python extension.
"""

from sys import platform
from Cython.Build import cythonize
import numpy as np

from setuptools import setup
from setuptools.extension import Extension
# Define the C++ extension
if platform == "darwin":
    extra_compile_args = ['-O3', '-pthread', '-funroll-loops', '-std=c++0x', '-stdlib=libc++', '-mmacosx-version-min=10.7']
else:
    extra_compile_args = ['-O3', '-pthread', '-funroll-loops', '-std=c++0x']

extensions = [
    Extension('*',
        sources=[
            'vectoria/FastTextLanguageModel.pyx',
        ],
        language='c++',
        extra_compile_args=extra_compile_args)
]

# Package details
setup(
    name='vectoria',
    version='0.0.1',
    author='Will Ballard',
    author_email='wballard@mailframe.net',
    url='https://github.com/wballard/vectoria',
    description='Word Vector Encoder',
    long_description=open('README.md', 'r').read(),
    license='BSD 3-Clause License',
    packages=['vectoria'],
    ext_modules = cythonize(extensions),
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
