import multiprocessing as mp
import numpy as np
import os
import queue

from .trac_ik_solver import TracIKSolver


class TracIKProc(mp.Process):
    """
    Used for finding ik in parallel.
    """

    def __init__(
        self,
        output_queue,
        urdf_file,
        base_link,
        tip_link,
        timeout=0.005,
        epsilon=1e-5,
        solve_type="Speed",
    ):
        super().__init__()
        self.output_queue = output_queue
        self.input_queue = mp.Queue()
        self.ik_solver = TracIKSolver(
            urdf_file,
            base_link,
            tip_link,
            timeout,
            epsilon,
            solve_type,
        )

    def _ik(self, ee_pose, qinit, bx, by, bz, brx, bry, brz):
        return self.ik_solver.ik(ee_pose, qinit, bx, by, bz, brx, bry, brz)

    def _fk(self, q):
        return self.ik_solver.fk(q)

    def run(self):
        while True:
            try:
                request = self.input_queue.get(timeout=1)
            except queue.Empty:
                continue
            ret = getattr(self, "_" + request[0])(*request[1:-1])
            self.output_queue.put((request[-1], ret))

    def ik(self, grasp, qinit, bx, by, bz, brx, bry, brz, ind=None):
        self.input_queue.put(
            ("ik", grasp, qinit, bx, by, bz, brx, bry, brz, ind)
        )

    def fk(self, q, ind=None):
        self.input_queue.put(("fk", q, ind))


class MultiTracIKSolver:
    def __init__(
        self,
        urdf_file,
        base_link,
        tip_link,
        timeout=0.005,
        epsilon=1e-5,
        solve_type="Speed",
        num_workers=os.cpu_count(),
    ):
        self.output_queue = mp.Queue()
        self.num_workers = num_workers
        self.ik_procs = []
        if (
            not isinstance(self.num_workers, int)
            or self.num_workers <= 0
            or self.num_workers > os.cpu_count()
        ):
            raise ValueError(
                "num_workers must be an integer between "
                f"1 and {os.cpu_count()}!"
            )
        for _ in range(num_workers):
            self.ik_procs.append(
                TracIKProc(
                    self.output_queue,
                    urdf_file,
                    base_link,
                    tip_link,
                    timeout,
                    epsilon,
                    solve_type,
                )
            )
            self.ik_procs[-1].daemon = True
            self.ik_procs[-1].start()

    @property
    def joint_limits(self):
        """
        Return lower bound limits and upper bound limits for all the joints
        in the order of the joint names.
        """
        return self.ik_procs[0].ik_solver.joint_limits

    @joint_limits.setter
    def joint_limits(self, bounds):
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

        for ikp in self.ik_procs:
            ikp.ik_solver.joint_limits = bounds

    @property
    def number_of_joints(self):
        return self.ik_procs[0].ik_solver.number_of_joints

    @property
    def joint_names(self):
        return self.ik_procs[0].ik_solver.joint_names

    @property
    def link_names(self):
        return self.ik_procs[0].ik_solver.link_names

    # Calculates FK for a vector of cfgs
    # (NOTE: this should be vectorized on C++ side)
    def fk(self, q):
        if not isinstance(q, np.ndarray):
            q = np.asarray(q, dtype=np.float64)
        if q.ndim == 1:
            q = q[None, :]
        if q.shape[1] != self.number_of_joints:
            raise ValueError(
                f"q must be of shape (n, {self.number_of_joints})!"
            )

        # Calculate FK for all cfgs distributed between processes
        for i, qi in enumerate(q):
            self.ik_procs[i % self.num_workers].fk(qi, ind=i)

        # collect computed fks
        fks = np.zeros((len(q), 4, 4))
        for _ in range(len(q)):
            output = self.output_queue.get(True)
            fks[output[0]] = output[1]

        return fks

    # Find ik for a single ee_pose and multiple seeds distributed
    # between processes. Returns the closest (according to norm)
    # to qinit if specified; otherwise the first found
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
        num_seeds=1,
        max_q_diffs=None,
        norm=2,
    ):
        if not isinstance(ee_pose, np.ndarray) or ee_pose.shape != (4, 4):
            raise ValueError("ee_pose must be numpy array of shape (4, 4)!")
        if qinit is not None and (
            not isinstance(qinit, np.ndarray)
            or qinit.shape != (self.number_of_joints,)
        ):
            raise ValueError(
                "qinit must either be None or numpy array of "
                f"shape ({self.number_of_joints},)!"
            )

        self.ik_procs[0].ik(
            ee_pose, qinit, bx=bx, by=by, bz=bz, brx=brx, bry=bry, brz=brz
        )
        output = self.output_queue.get(True)
        if output[1] is not None and (
            qinit is None
            or max_q_diffs is None
            or np.all(np.abs(output[1] - qinit) <= max_q_diffs)
        ):
            return output[1]
        for i in range(num_seeds - 1):
            self.ik_procs[(i + 1) % self.num_workers].ik(
                ee_pose,
                qinit=qinit,
                bx=bx,
                by=by,
                bz=bz,
                brx=brx,
                bry=bry,
                brz=brz,
            )

        # collect computed iks
        final_ik = []
        for _ in range(num_seeds - 1):
            output = self.output_queue.get(True)
            if output[1] is None:
                continue
            final_ik.append(output[1])

        final_ik = np.array(final_ik)
        if len(final_ik) == 0:
            return None
        elif qinit is None:
            return final_ik[0]
        else:
            # Filter ik greater than max diff
            valid_ik = (
                np.ones(len(final_ik), dtype=bool)
                if max_q_diffs is None
                else np.all(np.abs(final_ik - qinit) <= max_q_diffs, axis=1)
            )
            final_ik = final_ik[valid_ik]
            closest_ind = np.argmin(
                np.linalg.norm(final_ik - qinit, axis=1, ord=norm)
            )
            return final_ik[closest_ind]

    # Finds ik for many ee_pose, qinit pairs
    def iks(
        self,
        ee_poses,
        qinits=None,
        bx=1e-5,
        by=1e-5,
        bz=1e-5,
        brx=1e-3,
        bry=1e-3,
        brz=1e-3,
        num_seeds=1,
    ):
        if (
            not isinstance(ee_poses, np.ndarray)
            or ee_poses.ndim != 3
            or ee_poses.shape[1] != 4
            or ee_poses.shape[2] != 4
        ):
            raise ValueError(
                "ee_poses must be a numpy array of shape (n, 4, 4)!"
            )
        if qinits is None:
            qinits = [None] * len(ee_poses)
        elif (
            not isinstance(qinits, np.ndarray)
            or qinits.ndim != 2
            or qinits.shape != (len(ee_poses), self.number_of_joints)
        ):
            raise ValueError(
                "qinits must be a 2D numpy array of same length as ee_poses!"
            )

        iks = np.empty((len(ee_poses), self.number_of_joints))
        valid = np.zeros(len(ee_poses), dtype=bool)
        for i, (ee_pose, qinit) in enumerate(zip(ee_poses, qinits)):
            ret = self.ik(ee_pose, qinit=qinit, num_seeds=num_seeds)
            if ret is not None:
                valid[i] = True
                iks[i] = ret

        return valid, iks
