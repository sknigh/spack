# Copyright 2013-2018 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

from spack import *


class RMatr(RPackage):
    """Package matR (Metagenomics Analysis Tools for R) is an analysis
    client for the MG-RAST metagenome annotation engine, part of the US
    Department of Energy (DOE) Systems Biology Knowledge Base (KBase).
    Customized analysis and visualization tools securely access remote
    data and metadata within the popular open source R language and
    environment for statistical computing."""

    homepage = "https://github.com/MG-RAST/matR"
    url      = "https://cran.r-project.org/src/contrib/matR_0.9.1.tar.gz"
    list_url = "https://cran.r-project.org/src/contrib/Archive/matR/matR_0.9.tar.gz"

    version('0.9.1', sha256='554aeff37b27d0f17ddeb62b2e1004aa1f29190300e4946b1bec1d7c2bde82e3')
    version('0.9', 'e2be8734009f5c5b9c1f6b677a77220a')

    depends_on('r-mgraster', type=('build', 'run'))
    depends_on('r-biom-utils', type=('build', 'run'))
