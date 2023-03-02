#!/usr/bin/env python
# -*- coding: utf-8 -*-
import setuptools

with open('README.md') as f:
    readme = f.read()

exec(open('calibration/_version.py').read())

setuptools.setup(
    name='calibration',
    version=__version__,
    author='Tuomas Välimäki',
    author_email='tuomas.valimaki@tuni.fi',
    description='A package for motion-based sensor to sensor calibration',
    long_description=readme,
    long_description_content_type='text/markdown',
    url='https://github.com/tau-alma/trajectory_calibration',
    license='MIT',
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy',
        'matplotlib',
        'scipy',
        'casadi',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
    ]
)
