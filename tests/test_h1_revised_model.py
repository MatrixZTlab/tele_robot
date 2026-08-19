import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from teleop.robot_control.topstar_h1.config import H1RobotConfig
from teleop.robot_control.topstar_h1.joint_convention import (
    H1_ARM_COORDINATE_CONVENTION,
    H1_ARM_HARD_LOWER,
    H1_ARM_HARD_UPPER,
    H1_ARM_HOME_Q,
    H1_ARM_HOME_Q_BEFORE_2026_08_13_1659,
    H1_ARM_HOME_Q_BEFORE_RIGHT_ARM_2026_08_13_1121,
    H1_ARM_HOME_Q_2026_08_12,
    H1_ARM_PREVIOUS_HOME_Q,
    H1_ARM_HW_TO_MODEL_SIGN,
    H1_ARM_SAFE_LOWER,
    H1_ARM_SAFE_UPPER,
    H1_MUJOCO_URDF_FILENAME,
    H1_LEGACY_ARM_COORDINATE_CONVENTION,
    arm_positions_to_current,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = REPO_ROOT / "assets" / "topstar_h1"


def test_revised_mujoco_urdf_is_selected_and_self_contained():
    urdf_path = MODEL_DIR / H1_MUJOCO_URDF_FILENAME

    assert Path(H1RobotConfig().urdf_path) == urdf_path
    root = ET.parse(urdf_path).getroot()
    assert len(root.findall("link")) == 30
    assert len(root.findall("joint")) == 29
    for mesh in root.findall(".//mesh"):
        filename = mesh.attrib["filename"]
        assert not filename.startswith("package://")
        assert (MODEL_DIR / filename).is_file()


def test_revised_arm_convention_is_hardware_aligned():
    np.testing.assert_array_equal(H1_ARM_HW_TO_MODEL_SIGN, np.ones(14))
    np.testing.assert_allclose(H1_ARM_HARD_LOWER[:7], H1_ARM_HARD_LOWER[7:])
    np.testing.assert_allclose(H1_ARM_HARD_UPPER[:7], H1_ARM_HARD_UPPER[7:])
    assert np.all(H1_ARM_SAFE_LOWER > H1_ARM_HARD_LOWER)
    assert np.all(H1_ARM_SAFE_UPPER < H1_ARM_HARD_UPPER)


def test_xview_home_uses_left_then_right_order_and_hard_limits():
    expected = np.array(
        [
            -1.249638291135, -0.439753158332, 0.390569780011, -1.713093020710,
            -2.298982597312, -0.905319736302, -0.775379973491,
            1.257509726062, -0.420938508996, -0.402612551850, -1.700613916558,
            2.175273659931, -0.830759270657, 0.942512702662,
        ]
    )
    before_latest_adjustment = np.array(
        [
            -1.251523246728, -0.617270596553, 0.290911479722, -1.399317727786,
            -2.319420402853, -0.731397676341, -0.829869152738,
            0.988694114670, -0.655196601199, 0.185964831800, -1.372265124381,
            2.226341993844, -0.541313867506, 0.712425947372,
        ]
    )
    before_right_adjustment = np.array(
        [
            -1.251523246728, -0.617270596553, 0.290911479722, -1.399317727786,
            -2.319420402853, -0.731397676341, -0.829869152738,
            0.988833741010, -0.655091881444, 0.185964831800, -1.372282577673,
            2.200947453227, -0.541331320799, 0.749025501786,
        ]
    )
    expected_2026_08_12 = np.array(
        [
            -1.245100435080, -0.767246739177, 0.243124364803, -1.148479007690,
            -2.212990225066, -0.503894008343, -0.953019584759,
            0.986494999812, -0.585016911976, 0.187954507147, -1.333972600592,
            2.182796029007, -0.435773807638, 0.766408981136,
        ]
    )
    previous_expected = np.array(
        [
            -1.322488334, -0.800390542, 0.287473181, -1.509046578,
            -2.502104016, -0.817320235, -0.526391302,
            1.056622329, -0.701098761, 0.062639867, -1.608774691,
            2.476866555, -0.845629476, 0.410972679,
        ]
    )
    np.testing.assert_allclose(H1_ARM_HOME_Q, expected)
    np.testing.assert_allclose(
        H1_ARM_HOME_Q_BEFORE_2026_08_13_1659,
        before_latest_adjustment,
    )
    np.testing.assert_allclose(
        H1_ARM_HOME_Q_BEFORE_RIGHT_ARM_2026_08_13_1121,
        before_right_adjustment,
    )
    np.testing.assert_allclose(H1_ARM_HOME_Q_2026_08_12, expected_2026_08_12)
    np.testing.assert_allclose(H1_ARM_PREVIOUS_HOME_Q, previous_expected)
    assert np.all(H1_ARM_HOME_Q >= H1_ARM_HARD_LOWER)
    assert np.all(H1_ARM_HOME_Q <= H1_ARM_HARD_UPPER)


def test_legacy_replay_conversion_only_flips_left_j4_j6():
    legacy = np.arange(1.0, 15.0)
    current = arm_positions_to_current(
        legacy, H1_LEGACY_ARM_COORDINATE_CONVENTION
    )
    expected = legacy.copy()
    expected[[3, 5]] *= -1.0
    np.testing.assert_array_equal(current, expected)
    np.testing.assert_array_equal(
        arm_positions_to_current(current, H1_ARM_COORDINATE_CONVENTION),
        current,
    )


def test_controller_go_home_sends_revised_hardware_aligned_pose():
    from teleop.robot_control.topstar_h1.arm_controller import H1ArmController

    controller = object.__new__(H1ArmController)
    controller.home_head_q = np.array([0.0, -0.3])
    sent = {}
    controller.move_joints_timed = lambda joints, duration, head_q: sent.update(
        joints=np.asarray(joints), duration=duration, head_q=np.asarray(head_q)
    )

    controller.go_home()

    np.testing.assert_allclose(sent["joints"], H1_ARM_HOME_Q)
    assert sent["duration"] == 5.0
    np.testing.assert_allclose(sent["head_q"], controller.home_head_q)


def test_revised_urdf_arm_limits_match_shared_hard_envelope():
    root = ET.parse(MODEL_DIR / H1_MUJOCO_URDF_FILENAME).getroot()
    joints = {joint.attrib["name"]: joint for joint in root.findall("joint")}
    names = [
        *[f"Robot_Left_Hand_{suffix}_Joint" for suffix in ("base", 1, 2, 3, 4, 5, 6)],
        *[f"Robot_Right_Hand_{suffix}_Joint" for suffix in ("base", 1, 2, 3, 4, 5, 6)],
    ]
    for index, name in enumerate(names):
        limit = joints[name].find("limit")
        assert limit is not None
        # The vendor joint_defs.py carries more decimal places than the URDF;
        # tolerate only the URDF's small text-rounding gap.
        rounding_tolerance = 5e-5
        assert (
            float(limit.attrib["lower"])
            <= H1_ARM_HARD_LOWER[index] + rounding_tolerance
        )
        assert (
            float(limit.attrib["upper"])
            >= H1_ARM_HARD_UPPER[index] - rounding_tolerance
        )


def test_pinocchio_and_mujoco_fk_agree_in_revised_convention():
    from teleop.robot_control._base.control_mode import ControlMode
    from teleop.robot_control.topstar_h1.arm_ik import H1ArmIK
    from teleop.robot_control.vr_mujoco_relative_teleop import H1MuJoCoLMIK

    pinocchio_ik = H1ArmIK(
        H1RobotConfig(limit_mode="hard"),
        ControlMode.ARMS_ONLY,
        visualization="off",
    )
    mujoco_ik = H1MuJoCoLMIK.from_h1_assets(
        REPO_ROOT, arm_limit_mode="hard"
    )

    pin_left, pin_right = pinocchio_ik.get_dual_arm_ee_poses(
        np.rad2deg(H1_ARM_HOME_Q)
    )
    mj_left, mj_right = mujoco_ik.forward_kinematics(H1_ARM_HOME_Q)

    np.testing.assert_allclose(pin_left, mj_left, atol=1e-8)
    np.testing.assert_allclose(pin_right, mj_right, atol=1e-8)

    pin_solution = pinocchio_ik.solve_ik(
        pin_left,
        pin_right,
        H1_ARM_HOME_Q,
        np.zeros(14),
        head_target=None,
    )
    mj_solution = mujoco_ik.solve(
        mj_left,
        mj_right,
        H1_ARM_HOME_Q,
        current_q_weight=0.0001,
    )
    np.testing.assert_allclose(pin_solution.arm_q, H1_ARM_HOME_Q, atol=1e-3)
    np.testing.assert_allclose(mj_solution.arm_q, H1_ARM_HOME_Q, atol=1e-8)

    solved_pin_left, solved_pin_right = pinocchio_ik.get_dual_arm_ee_poses(
        np.rad2deg(pin_solution.arm_q)
    )
    np.testing.assert_allclose(solved_pin_left, pin_left, atol=5e-4)
    np.testing.assert_allclose(solved_pin_right, pin_right, atol=5e-4)
