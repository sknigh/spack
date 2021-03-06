# Copyright 2013-2018 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

from spack import *


class PyKeras(PythonPackage):
    """Deep Learning library for Python. Convnets, recurrent neural networks,
    and more. Runs on Theano or TensorFlow."""

    homepage = "http://keras.io"
    url      = "https://pypi.io/packages/source/K/Keras/Keras-1.2.2.tar.gz"

    version('2.0.3', '39ce72a65623cd233a8fa4e867dd0c6b')
    version('1.2.2', '8e26b25bf16494f6eca726887d232319')
    version('1.2.1', '95525b9faa890267d80d119b13ce2984')
    version('1.2.0', 'd24d8b72747f8cc38e659ce8fc92ad3c')
    version('1.1.2', '53027097f240735f873119ee2e8d27ff')
    version('1.1.1', '4bd8b75e8c6948ec0498cc603bbc6590')
    version('1.1.0', 'd1711362ac8473238b0d198d2e3a0574')

    depends_on('py-setuptools', type='build')
    depends_on('py-theano', type=('build', 'run'))
    depends_on('py-pyyaml', type=('build', 'run'))
    depends_on('py-six', type=('build', 'run'))
