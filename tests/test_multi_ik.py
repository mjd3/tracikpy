import numpy as np
import os
import pytest

from tracikpy import MultiTracIKSolver


@pytest.fixture
def ik_solver(num_workers):
    if num_workers is not None:
        return MultiTracIKSolver(
            "data/franka_panda.urdf",
            "panda_link0",
            "panda_hand",
            timeout=0.05,
            num_workers=num_workers,
        )
    else:
        return MultiTracIKSolver(
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


# Test initialization
@pytest.mark.parametrize("num_workers", [None, 2])
def test_init(ik_solver):
    assert len(ik_solver.ik_procs) == ik_solver.num_workers
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


# Test bad initialization
def test_bad_init():
    with pytest.raises(ValueError):
        MultiTracIKSolver(
            "data/franka_panda.urdf",
            "panda_link0",
            "panda_hand",
            timeout=0.05,
            num_workers=2.5,
        )

    with pytest.raises(ValueError):
        MultiTracIKSolver(
            "data/franka_panda.urdf",
            "panda_link0",
            "panda_hand",
            timeout=0.05,
            num_workers=0,
        )

    with pytest.raises(ValueError):
        MultiTracIKSolver(
            "data/franka_panda.urdf",
            "panda_link0",
            "panda_hand",
            timeout=0.05,
            num_workers=os.cpu_count() + 1,
        )


# Test getting and setting joint limits
@pytest.mark.parametrize("num_workers", [None])
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


# Test solving FK in parallel
@pytest.mark.parametrize("num_workers", [None])
def test_multi_fk(ik_solver):
    rand_qs = np.random.default_rng().uniform(
        *ik_solver.joint_limits, size=(100, ik_solver.number_of_joints)
    )
    output_fks = ik_solver.fk(rand_qs)
    assert output_fks.shape == (100, 4, 4)
    output_fks = ik_solver.fk(rand_qs.tolist())
    assert output_fks.shape == (100, 4, 4)


@pytest.mark.parametrize("num_workers", [None])
def test_multi_ik(ik_solver, ee_pose):
    # Test solving IK in parallel without qinit
    qout = ik_solver.ik(ee_pose, num_seeds=100)
    assert qout is not None
    assert qout.shape == (ik_solver.number_of_joints,)

    # Test solving IK in parallel with qinit
    qout = ik_solver.ik(
        ee_pose, qinit=np.zeros(ik_solver.number_of_joints), num_seeds=100
    )
    assert qout is not None
    assert qout.shape == (ik_solver.number_of_joints,)

    # Test case where no solution is available (unreachable pose)
    bad_ee_pose = ee_pose.copy()
    bad_ee_pose[2, 3] += 2
    qout = ik_solver.ik(
        bad_ee_pose, qinit=np.zeros(ik_solver.number_of_joints)
    )
    assert qout is None


@pytest.mark.parametrize("num_workers", [None])
def test_multi_iks(ik_solver, ee_pose):
    # Test iks with no qinit
    valid, qouts = ik_solver.iks(
        np.repeat(ee_pose[None, ...], 100, axis=0), num_seeds=100
    )
    assert qouts.shape == (100, ik_solver.number_of_joints)
    assert valid.all()

    # Test iks with qinit
    valid, qouts = ik_solver.iks(
        np.repeat(ee_pose[None, ...], 100, axis=0),
        qinits=np.zeros((100, ik_solver.number_of_joints)),
        num_seeds=100,
    )
    assert qouts.shape == (100, ik_solver.number_of_joints)
    assert valid.all()


# Test that exceptions are raised correctly for bad inputs
@pytest.mark.parametrize("num_workers", [None])
def test_bad_inputs(ik_solver, ee_pose):
    bad_qinit = np.zeros(ik_solver.number_of_joints - 1)
    with pytest.raises(ValueError):
        ik_solver.ik(ee_pose, qinit=bad_qinit)

    with pytest.raises(ValueError):
        ik_solver.ik(ee_pose[:3, :3])

    with pytest.raises(ValueError):
        ik_solver.iks(ee_pose[None, :3, :3])

    with pytest.raises(ValueError):
        ik_solver.iks(
            np.repeat(ee_pose[None, ...], 100, axis=0),
            np.zeros((99, ik_solver.number_of_joints)),
        )

    with pytest.raises(ValueError):
        ik_solver.fk(bad_qinit)

    bounds = ik_solver.joint_limits[0]
    with pytest.raises(ValueError):
        ik_solver.joint_limits = bounds

    with pytest.raises(ValueError):
        ik_solver.joint_limits = (bad_qinit, bounds)

    with pytest.raises(ValueError):
        ik_solver.joint_limits = (bounds, bad_qinit)
