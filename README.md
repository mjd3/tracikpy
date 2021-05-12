# Tracikpy
This repo contains Python bindings for TracIK based on the ROS packages and SWIG bindings provided by TRACLabs [here](https://bitbucket.org/traclabs/trac_ik/src/master/) (see [paper](https://ieeexplore.ieee.org/document/7363472) for implementation details) and the updates provided by Clemens Eppner [here](https://bitbucket.org/clemi/trac_ik/src/devel/). The goal of this repo is to encapsulate these bindings into a Python package that can be easily installed using `pip` and can be used independently of ROS. This package does still contain several system-level dependencies (`eigen`, `orocos-kdl`, `nlopt`, `urdf`, and `kdl_parser`), but the Python wrapper relies only on `numpy`.

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
pip install tracikpy
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
assert np.linalg.norm(ee_diff[:3, 3], ord=1) < 1e-3
assert np.linalg.norm(ee_diff[:3, :3] - np.eye(3), ord=1) < 1e-3
```
which should not output any assertion errors since the pose is close to the desired pose.

## TODO
 - Fix GitHub Actions CI (figure out how to include external deps)
 - Migrate SWIG std_vector.i templates to numpy.i (see [here](https://numpy.org/devdocs/reference/swig.interface-file.html))
 - Integrate CIBuildWheel?

Many thanks to TRACLabs (Patrick Beeson and Barrett Ames) for the initial TracIK library, to Sammy Pfeiffer for the initial Python bindings, and to Clemens Eppner for initial cleanup to remove ROS dependencies.