#!/usr/bin/env python
import os

from setuptools import Extension, find_packages, setup
from setuptools.command.build_py import build_py as _build_py


# Build extensions before python modules,
# or the generated SWIG python files will be missing.
class build_py(_build_py):
    def run(self):
        self.run_command("build_ext")
        super(_build_py, self).run()


# load __version__ without importing anything
version_file = os.path.join(os.path.dirname(__file__), "tracikpy/version.py")
with open(version_file, "r") as f:
    # use eval to get a clean string of version from file
    __version__ = eval(f.read().strip().split("=")[-1])

_trac_ik_wrap = Extension(
    "tracikpy.swig._trac_ik_wrap",
    [
        "tracikpy/swig/trac_ik.i",
        "tracikpy/src/trac_ik.cpp",
        "tracikpy/src/nlopt_ik.cpp",
        "tracikpy/src/kdl_tl.cpp",
    ],
    include_dirs=[
        "tracikpy/include",
        "/usr/include/eigen3",
    ],
    libraries=["orocos-kdl", "nlopt", "urdf", "kdl_parser"],
    swig_opts=["-c++", "-Itracikpy/include"],
    extra_compile_args=["-std=c++11"],
)

setup(
    name="tracikpy",
    version=__version__,
    description="TracIK Python Bindings",
    author="Michael Danielczuk",
    author_email="mdanielczuk@berkeley.edu",
    license="MIT Software License",  # Probably need to change this
    url="https://github.com/mjd3/tracikpy",
    keywords="robotics inverse kinematics",
    classifiers=[
        "License :: OSI Approved :: MIT Software License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Natural Language :: English",
        "Topic :: Scientific/Engineering",
    ],
    packages=find_packages(),
    cmdclass={"build_py": build_py},
    ext_modules=[_trac_ik_wrap],
    install_requires=["numpy"],
    extras_require={"test": ["pytest", "pytest-cov"]},
)
