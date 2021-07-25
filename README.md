# Tracikpy
[![status](https://github.com/mjd3/tracikpy/workflows/Release%20Tracikpy/badge.svg)](https://github.com/mjd3/tracikpy/actions) [![Coverage Status](https://coveralls.io/repos/github/mjd3/tracikpy/badge.svg?branch=main)](https://coveralls.io/github/mjd3/tracikpy?branch=main) [![style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This repo contains Python bindings for TracIK based on the ROS packages and SWIG bindings provided by TRACLabs [here](https://bitbucket.org/traclabs/trac_ik/src/master/) (see [paper](https://ieeexplore.ieee.org/document/7363472) for implementation details) and the updates provided by Clemens Eppner [here](https://bitbucket.org/clemi/trac_ik/src/devel/). For now, it only supports Ubuntu operating systems. The goal of this repo is to encapsulate these bindings into a Python package that can be easily installed using `pip` (with `numpy` as the only Python dependency) and can be used independently of ROS. This package does still contain several system-level dependencies (`eigen`, `orocos-kdl`, `nlopt`, `urdfdom`, and `kdl_parser`); if you have already installed ROS then these will be installed already. If not, you can install using `apt-get`:
```
sudo apt-get install libeigen3-dev liborocos-kdl-dev libkdl-parser-dev liburdfdom-dev libnlopt-dev
```
For Ubuntu 20.04, `libnlopt-cxx-dev` is also needed to install the C++ bindings for `nlopt`.

Main differences from original library/bindings:
 - ROS and Boost dependencies removed (replaced with C++ standard library calls).
 - Installable via `pip`, taking advantage of SWIG support in `setuptools.Extension`.
 - Added forward kinematics.
 - Simplified Python wrapper.
 - Added basic unit tests through `pytest`

## Install
Clone the repo and install using `pip`:
```shell
git clone https://github.com/mjd3/tracikpy.git
pip install tracikpy/
```
That's it!

## Quickstart

Here is an example script showing how this package can be used to calculate inverse kinematics:
```python
import numpy as np
from tracikpy import TracIKSolver

ee_pose = np.array([[ 0.0525767 , -0.64690764, -0.7607537 , 0.        ],
                    [-0.90099786, -0.35923817,  0.24320937, 0.2       ],
                    [-0.43062577,  0.67265031, -0.60174996, 0.4       ],
                    [ 0.        ,  0.        ,  0.        , 1.        ]])

ik_solver = TracIKSolver(
    "data/franka_panda.urdf",
    "panda_link0",
    "panda_hand",
)
qout = ik_solver.ik(ee_pose, qinit=np.zeros(ik_solver.number_of_joints))
print(qout)
```
which yields:
```
[-0.05331206 -1.75956216  0.6304369  -3.00941705  1.28849325  0.46002026  2.58126884]
```
Note that the `qinit` argument is optional; if you do not include it then it will be set to a configuration chosen uniformly at random between the robot's joint limits.

You can also check the solution using forward kinematics:
```python
ee_out = ik_solver.fk(qout)
ee_diff = np.linalg.inv(ee_pose) @ ee_out
trans_err = np.linalg.norm(ee_diff[:3, 3], ord=1)
angle_err = np.arccos(np.trace(ee_diff[:3, :3] - 1) / 2)
assert trans_err < 1e-3
assert angle_err < 1e-3 or angle_err - np.pi < 1e-3
```
which should not output any assertion errors since the pose is close to the desired pose.

## Unit Tests
To run the unit tests for this project, install the repo with the `test` option and run `pytest`:
```shell
cd /path/to/tracikpy
pip install -e .[test]
pytest
```

## TODO
 - GitHub Actions CI for MacOS and Windows
 - Migrate SWIG std_vector.i templates to numpy.i (see [here](https://numpy.org/devdocs/reference/swig.interface-file.html)) or migrate to pybind11.
 - Integrate CIBuildWheel?

## Acknowledgments
Many thanks to TRACLabs (Patrick Beeson and Barrett Ames) for the initial TracIK library, to Sammy Pfeiffer for the initial Python bindings, and to Clemens Eppner for initial cleanup to remove ROS dependencies.
