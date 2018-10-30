# Copyright 2013-2018 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

from spack import *


class SstMacro(AutotoolsPackage):
    """The Structural Simulation Toolkit Macroscale Element Library simulates
    large-scale parallel computer architectures for the coarse-grained study
    of distributed-memory applications. The simulator is driven from either a
    trace file or skeleton application. SST/macro's modular architecture can
    be extended with additional network models, trace file formats, software
    services, and processor models.
    """

    homepage = "http://sst.sandia.gov/about_sstmacro.html"
    url      = "https://github.com/sstsimulator/sst-macro/releases/download/v6.1.0_Final/sstmacro-6.1.0.tar.gz"
    git      = "https://github.com/sstsimulator/sst-macro.git"

    version('develop', branch='devel')
    version('8.0.0', sha256='8618a259e98ede9a1a2ce854edd4930628c7c5a770c3915858fa840556c1861f')
    version('6.1.0', '98b737be6326b8bd711de832ccd94d14')

    depends_on('boost@1.59:', when='@:6.1.0')

    depends_on('autoconf@1.68:', type='build', when='@develop')
    depends_on('automake@1.11.1:', type='build', when='@develop')
    depends_on('libtool@1.2.4:', type='build', when='@develop')
    depends_on('m4', type='build', when='@develop')

    depends_on('binutils', type='build')
    depends_on('llvm+clang@:5', when='@:8.0.0+skeletonizer')
    depends_on('llvm+clang', when='@8.1.0:+skeletonizer')
    depends_on('mpi', when='+mpi')
    depends_on('otf2', when='+otf2')
    depends_on('sst-core@8.0.0', when='@8.0.0 +core')
    depends_on('sst-core@develop', when='@develop +core')
    depends_on('vtk@8.1.0:~haru+osmesa', when='+vtk')
    depends_on('zlib', type=('build', 'link'))

    # VTK is not available before v8.1.0
    conflicts('+vtk', when='@:8.0.0')

    variant('core', default=False, description='Use SST Core for PDES')
    variant('debug', default=False, description='Build with debug flags and link to address sanitizer')
    variant('mpi', default=True, description='Enable distributed PDES simulation')
    variant('otf2', default=False, description='Enable OTF2 trace emission and replay support')
    variant('shared', default=True, description='Build shared libraries')
    variant('skeletonizer', default=False, description='Enable Clang source-to-source autoskeletonization')
    variant('static', default=True, description='Build static libraries')
    variant('threaded', default=False, description='Enable thread-parallel PDES simulation')
    variant('vtk', default=False, description='Enable VTK visualization support ')

    @run_before('autoreconf')
    def bootstrap(self):
        if '@develop' in self.spec:
            Executable('./bootstrap.sh')()

    def configure_args(self):
        args = ['--disable-regex']

        spec = self.spec
        if '+debug' in spec:
            flags = '-g -O1 -fsanitize=address -fsanitize-recover=address'
            env['CFLAGS'] = flags
            env['CXXFLAGS'] = flags
            env['LDFLAGS'] = '-fsanitize=address'
        else:
            env['CFLAGS'] = '-O2'
            env['CXXFLAGS'] = '-O2'

        args.append(
            '--enable-static=%s' % ('yes' if '+static' in spec else 'no'))
        args.append(
            '--enable-shared=%s' % ('yes' if '+shared' in spec else 'no'))

        if spec.satisfies("@8.0.0:"):
            args.extend([
                '--%sable-otf2' % ('en' if '+otf2' in spec else 'dis'),
                '--%sable-multithread' % (
                    'en' if '+threaded' in spec else 'dis')
            ])

            if '+skeletonizer' in spec:
                args.append('--with-clang=' + spec['llvm'].prefix)

        if spec.satisfies("@8.1.0:"):
            # Optional VTK support
            if '+vtk' in spec:
                args.extend([
                    '--with-vtk=%s' % self.spec['vtk'].prefix,
                    # VTK obnoxiously suffixes paths with a major.minor string
                    '--enable-vtk=%s' % self.spec['vtk'].version.up_to(2),
                ])

        if '+core' in spec:
            args.append('--with-sst-core=%s' % spec['sst-core'].prefix)

        # Optional MPI support
        if '+mpi' in spec:
            env['CC'] = spec['mpi'].mpicc
            env['CXX'] = spec['mpi'].mpicxx
            env['F77'] = spec['mpi'].mpif77
            env['FC'] = spec['mpi'].mpifc

        return args
