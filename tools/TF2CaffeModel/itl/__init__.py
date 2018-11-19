#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Deep learning and Reinforcement learning library for Researchers and Engineers"""

from __future__ import absolute_import

import os

if 'itl_PACKAGE_BUILDING' not in os.environ:

    try:
        import tensorflow
    except Exception as e:
        raise ImportError(
            "Tensorflow is not installed, please install it with the one of the following commands:\n"
            " - `pip install --upgrade tensorflow`\n"
            " - `pip install --upgrade tensorflow-gpu`"
        )
    tf_version_major = int(tensorflow.__version__.split('.')[0])
    tf_version_minor = int(tensorflow.__version__.split('.')[1])
    if (tf_version_major < 1) or (tf_version_major == 1 and tf_version_minor < 6) and os.environ.get('READTHEDOCS', None) != 'True':
        raise RuntimeError(
            "itl does not support Tensorflow version older than 1.6.0.\n"
            "Please update Tensorflow with:\n"
            " - `pip install --upgrade tensorflow`\n"
            " - `pip install --upgrade tensorflow-gpu`"
        )

    # from itl.lazy_imports import LazyImport

    # from itl import activation
    # from itl import array_ops
    # from itl import cost
    # from itl import db
    # from itl import decorators
    # from itl import files
    # from itl import initializers
    # from itl import iterate
    from itl import layers
    # from itl import lazy_imports
    # from itl import logging
    # from itl import models
    # from itl import optimizers
    # from itl import rein
    # from itl import utils

    # Lazy Imports
    # distributed = LazyImport("itl.distributed")
    # nlp = LazyImport("itl.nlp")
    # prepro = LazyImport("itl.prepro")
    # visualize = LazyImport("itl.visualize")

    # alias
    # act = activation
    # vis = visualize

    # alphas = array_ops.alphas
    # alphas_like = array_ops.alphas_like

    # global vars
    global_flag = {}
    global_dict = {}

# Use the following formatting: (major, minor, patch, prerelease)
VERSION = (1, 9, 0, "")
__shortversion__ = '.'.join(map(str, VERSION[:3]))
__version__ = '.'.join(map(str, VERSION[:3])) + "".join(VERSION[3:])

__package_name__ = 'itl'
__contact_names__ = 'itl Contributors'
__contact_emails__ = 'hao.dong11@imperial.ac.uk'
__homepage__ = 'http://itl.readthedocs.io/en/latest/'
__repository_url__ = 'https://github.com/itl/itl'
__download_url__ = 'https://github.com/itl/itl'
__description__ = 'Reinforcement Learning and Deep Learning Library for Researcher and Engineer.'
__license__ = 'apache'
__keywords__ = 'deep learning, machine learning, computer vision, nlp, supervised learning, unsupervised learning, reinforcement learning, tensorflow'
