import numpy as np
import pytest

from tracikpy import TracIKSolver


@pytest.fixture
def ik_solver():
    return TracIKSolver(
        "data/franka_panda.urdf", "panda_link0", "panda_hand", timeout=0.05
    )


@pytest.fixture
def ee_pose():
    return np.array(
        [
            [0.0525767, -0.64690764, -0.7607537, 0.0],
            [-0.90099786, -0.35923817, 0.24320937, 0.2],
            [-0.43062577, 0.67265031, -0.60174996, 0.4],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


# Test URDF loading and initialization
def test_init(ik_solver):
    assert ik_solver.number_of_joints == 7
    assert ik_solver.joint_names == (
        "panda_joint1",
        "panda_joint2",
        "panda_joint3",
        "panda_joint4",
        "panda_joint5",
        "panda_joint6",
        "panda_joint7",
    )
    assert ik_solver.link_names == (
        "panda_link1",
        "panda_link2",
        "panda_link3",
        "panda_link4",
        "panda_link5",
        "panda_link6",
        "panda_link7",
        "panda_hand",
    )


# Test getting and setting joint limits
def test_joint_limits(ik_solver):
    assert np.allclose(
        ik_solver.joint_limits[0],
        np.array(
            [
                -2.8973,
                -1.76279998,
                -2.8973,
                -3.07179999,
                -2.8973,
                -0.0175,
                -2.8973,
            ]
        ),
    )
    assert np.allclose(
        ik_solver.joint_limits[1],
        np.array(
            [2.8973, 1.76279998, 2.8973, -0.0698, 2.8973, 3.75250006, 2.8973]
        ),
    )

    rg = np.random.default_rng()
    new_limits = (
        rg.standard_normal(ik_solver.number_of_joints),
        rg.standard_normal(ik_solver.number_of_joints),
    )
    ik_solver.joint_limits = new_limits
    assert np.allclose(ik_solver.joint_limits[0], new_limits[0])
    assert np.allclose(ik_solver.joint_limits[1], new_limits[1])


# Test IK solutions to make sure they match desired pose
def test_ik_fk(ik_solver, ee_pose):
    qout = ik_solver.ik(ee_pose, qinit=np.zeros(ik_solver.number_of_joints))
    ee_out = ik_solver.fk(qout)
    ee_diff = np.linalg.inv(ee_pose) @ ee_out
    trans_err = np.linalg.norm(ee_diff[:3, 3], ord=1)
    angle_err = np.arccos(np.trace(ee_diff[:3, :3] - 1) / 2)
    assert trans_err < 1e-3
    assert angle_err < 1e-3 or angle_err - np.pi < 1e-3

    # Try with lists (note: these can't be checked against previous outputs as
    # solution may change)
    ik_solver.ik(
        ee_pose.tolist(), qinit=np.zeros(ik_solver.number_of_joints).tolist()
    )
    ee_out_list = ik_solver.fk(qout.tolist())
    ee_diff = np.linalg.inv(ee_pose) @ ee_out_list
    trans_err = np.linalg.norm(ee_diff[:3, 3], ord=1)
    angle_err = np.arccos(np.trace(ee_diff[:3, :3] - 1) / 2)
    assert trans_err < 1e-3
    assert angle_err < 1e-3 or angle_err - np.pi < 1e-3

    # Test random initializations (no qinit specified)
    # Can't check results here since it may be None depending on qinit
    qout = ik_solver.ik(ee_pose)

    # Test case where no solution is available (unreachable pose)
    bad_ee_pose = ee_pose.copy()
    bad_ee_pose[2, 3] += 2
    qout = ik_solver.ik(
        bad_ee_pose, qinit=np.zeros(ik_solver.number_of_joints)
    )
    assert qout is None


# Test that exceptions are raised correctly
def test_ik_errs(ik_solver, ee_pose):
    bad_qinit = np.zeros(ik_solver.number_of_joints - 1)
    with pytest.raises(ValueError):
        ik_solver.ik(ee_pose, qinit=bad_qinit)

    with pytest.raises(ValueError):
        ik_solver.fk(bad_qinit)

    bounds = ik_solver.joint_limits[0]
    with pytest.raises(ValueError):
        ik_solver.joint_limits = bounds

    with pytest.raises(ValueError):
        ik_solver.joint_limits = (bad_qinit, bounds)

    with pytest.raises(ValueError):
        ik_solver.joint_limits = (bounds, bad_qinit)
