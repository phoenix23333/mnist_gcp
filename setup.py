# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 14:39:14 2018

@author: wz
"""

from setuptools import find_packages
from setuptools import setup
import logging
import subprocess
from setuptools.command.install import install



setup(
    name='trainer',
    version='0.1',
    packages=find_packages(),
    include_package_data=True,
    description='My trainer application package.',
	install_requires=[
      'keras',
      'h5py'

  ]

)