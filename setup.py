'''
Author(s): Craig Fouts
Correspondence: c.fouts25@imperial.ac.uk
License: Apache 2.0 license
'''

from setuptools import find_packages, setup, Extension

setup(
    name='terrarium',
    version='0.0.1',
    description='Experiments with Lotka-Volterra models.',
    author='Craig Fouts',
    author_email='c.fouts25@imperial.ac.uk',
    packages=find_packages(),
    ext_modules=[]
)
