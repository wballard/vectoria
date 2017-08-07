from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize
from sys import platform
import unittest

# Define the C++ extension
if platform == "darwin":
    extra_compile_args = ['-O3', '-pthread', '-funroll-loops', '-std=c++11', '-stdlib=libc++', '-mmacosx-version-min=10.7']
else:
    extra_compile_args = ['-O3', '-pthread', '-funroll-loops', '-std=c++11']

extensions = [
    Extension('vecoder.fasttext',
        sources=[
            'vecoder/fasttext.pyx',
            'vecoder/fasttext/args.cc',
            'vecoder/fasttext/dictionary.cc',
            'vecoder/fasttext/matrix.cc',
            'vecoder/fasttext/model.cc',
            'vecoder/fasttext/utils.cc',
            'vecoder/fasttext/fasttext.cc',
            'vecoder/fasttext/vector.cc',
            'vecoder/fasttext/main.cc'
        ],
        language='c++',
        extra_compile_args=extra_compile_args)
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
    ext_modules = cythonize(extensions),
    install_requires=[
        'numpy>=1',
        'future'
    ],
    classifiers= [
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
