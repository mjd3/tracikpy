#!/usr/bin/env python
from setuptools import setup, Extension
from setuptools.command.build_py import build_py as _build_py


class build_py(_build_py):
    def run(self):
        self.run_command("build_ext")
        return super().run()


setup(
    name='tracikpy',
    version='0.1',
    cmdclass={'build_py': build_py},
    packages=["tracikpy"],
    ext_modules=[
        Extension(
            'tracikpy._tracik',
            [
                'tracikpy/trac_ik.i',
                'tracikpy/trac_ik.cpp',
                'tracikpy/nlopt_ik.cpp',
                'tracikpy/kdl_tl.cpp',
            ],
            include_dirs=[
                "tracikpy",
                "/usr/include/eigen3",
            ],
            libraries=["orocos-kdl", "nlopt", "urdf", "kdl_parser"],
            swig_opts=['-c++'],
        )
    ],
)
