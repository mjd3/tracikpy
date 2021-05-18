#!/usr/bin/env python

# Author: Mike Danielczuk <mdanielczuk@berkeley.edu>
# Adapted from Sammy Pfeiffer <Sammy.Pfeiffer at student.uts.edu.au>
# Convenience code to wrap TRAC IK

import numpy as np

from .swig.trac_ik_wrap import TRAC_IK


class TracIKSolver:
    def __init__(
        self,
        urdf_file,
        base_link,
        tip_link,
        timeout=0.005,
        epsilon=1e-5,
        solve_type="Speed",
    ):
        """
        Create a TRAC_IK instance and keep track of it.

        :param urdf_file str: path to URDF file
        :param str base_link: Starting link of the chain.
        :param str tip_link: Last link of the chain.
        :param float timeout: Timeout in seconds for the IK calls.
        :param float epsilon: Error epsilon.
        :param solve_type str: Type of solver, can be:
            Speed (default), Distance, Manipulation1, Manipulation2

        """
        self._urdf_string = "".join(open(urdf_file, "r").readlines())
        self.base_link = base_link
        self.tip_link = tip_link
        self._timeout = timeout
        self._epsilon = epsilon
        self._solve_type = solve_type
        self._ik_solver = TRAC_IK(
            self.base_link,
            self.tip_link,
            self._urdf_string,
            self._timeout,
            self._epsilon,
            self._solve_type,
        )
        self.number_of_joints = self._ik_solver.getNrOfJointsInChain()
        self.joint_names = self._ik_solver.getJointNamesInChain(
            self._urdf_string
        )
        self.link_names = self._ik_solver.getLinkNamesInChain()

    def ik(
        self,
        ee_pose,
        qinit=None,
        bx=1e-5,
        by=1e-5,
        bz=1e-5,
        brx=1e-3,
        bry=1e-3,
        brz=1e-3,
    ):
        """
        Do the IK call.

        :param list of list of float ee_pose: desired tip link pose.
        :param list of float qinit: Initial joint configuration as seed.
        :param float bx: X allowed bound.
        :param float by: Y allowed bound.
        :param float bz: Z allowed bound.
        :param float brx: rotation over X allowed bound.
        :param float bry: rotation over Y allowed bound.
        :param float brz: rotation over Z allowed bound.

        :return: joint values or None if no solution found.
        :rtype: np.ndarray of float.
        """
        if qinit is None:
            qinit = np.random.default_rng().uniform(*self.joint_limits)
        elif len(qinit) != self.number_of_joints:
            raise ValueError(
                f"qinit has length {len(qinit):d} "
                f"and it should have length {self.number_of_joints:d}"
            )

        if not isinstance(ee_pose, np.ndarray) or ee_pose.dtype != np.float64:
            ee_pose = np.array(ee_pose, dtype=np.float64)
        if not isinstance(qinit, np.ndarray) or qinit.dtype != np.float64:
            qinit = np.array(qinit, dtype=np.float64)
        xyz = ee_pose[:3, 3]
        q = self._q_from_pose(ee_pose)
        q = np.roll(q, -1)
        solution = self._ik_solver.CartToJnt(
            qinit, *xyz, *q, bx, by, bz, brx, bry, brz
        )
        return np.array(solution) if solution else None

    def fk(self, q):
        if len(q) != self.number_of_joints:
            raise ValueError(
                f"q has length {len(q):d} "
                f"and it should have length {self.number_of_joints:d}"
            )
        if not isinstance(q, np.ndarray) or q.dtype != np.float64:
            q = np.array(q, dtype=np.float64)
        solution = self._ik_solver.JntToCart(q)
        return np.array(solution) if solution else None

    @property
    def joint_limits(self):
        """
        Return lower bound limits and upper bound limits for all the joints
        in the order of the joint names.
        """
        lb = self._ik_solver.getLowerBoundLimits()
        ub = self._ik_solver.getUpperBoundLimits()
        return np.array(lb), np.array(ub)

    @joint_limits.setter
    def joint_limits(self, bounds):
        """
        Set joint limits for all the joints.

        :arg list lower_bounds: List of float of the lower bound limits for
            all joints.
        :arg list upper_bounds: List of float of the upper bound limits for
            all joints.
        """
        try:
            lower_bounds, upper_bounds = bounds
        except ValueError:
            raise ValueError("bounds must be an iterable with two lists")
        if len(lower_bounds) != self.number_of_joints:
            raise ValueError(
                "lower_bounds array size mismatch, input size "
                f"{len(lower_bounds):d}, should be {self.number_of_joints:d}"
            )

        if len(upper_bounds) != self.number_of_joints:
            raise ValueError(
                "upper_bounds array size mismatch, input size "
                f"{len(upper_bounds):d}, should be {self.number_of_joints:d}"
            )
        self._ik_solver.setKDLLimits(lower_bounds, upper_bounds)

    def _q_from_pose(self, pose):
        q = np.empty((4,))
        t = np.trace(pose)
        if t > pose[3, 3]:
            q[0] = t
            q[3] = pose[1, 0] - pose[0, 1]
            q[2] = pose[0, 2] - pose[2, 0]
            q[1] = pose[2, 1] - pose[1, 2]
        else:
            i, j, k = 0, 1, 2
            if pose[1, 1] > pose[0, 0]:
                i, j, k = 1, 2, 0
            if pose[2, 2] > pose[i, i]:
                i, j, k = 2, 0, 1
            t = pose[i, i] - (pose[j, j] + pose[k, k]) + pose[3, 3]
            q[i] = t
            q[j] = pose[i, j] + pose[j, i]
            q[k] = pose[k, i] + pose[i, k]
            q[3] = pose[k, j] - pose[j, k]
            q = q[[3, 0, 1, 2]]
        q *= 0.5 / np.sqrt(t * pose[3, 3])
        if q[0] < 0.0:
            np.negative(q, q)
        return q
