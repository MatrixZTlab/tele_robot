import numpy as np
import pytest

from teleop.robot_control._base.dual_object_control import DualObjectController
from teleop.robot_control._base.robot_driver import RobotDriver
from teleop.robot_control._base.xr_transformer import XRProcessedData


def _pose(position=(0.0, 0.0, 0.0), rotation=None):
    result = np.eye(4)
    result[:3, 3] = np.asarray(position, dtype=float)
    if rotation is not None:
        result[:3, :3] = rotation
    return result


def _rotation_z(angle):
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _controller():
    return DualObjectController(
        max_translation_speed_m_s=100.0,
        max_rotation_speed_rad_s=100.0,
    )


def _lock(controller):
    input_left = _pose((-0.25, 0.0, 0.0))
    input_right = _pose((0.25, 0.0, 0.0))
    robot_left = _pose((0.30, 0.20, 1.00))
    robot_right = _pose((0.30, -0.20, 1.00))
    width = controller.lock(
        input_left, input_right, robot_left, robot_right, now_s=0.0
    )
    return input_left, input_right, robot_left, robot_right, width


def test_lock_starts_from_measured_robot_poses():
    controller = _controller()
    input_left, input_right, robot_left, robot_right, width = _lock(controller)

    left, right = controller.targets(input_left, input_right, now_s=0.1)

    assert controller.locked
    assert width == pytest.approx(0.4)
    np.testing.assert_allclose(left, robot_left)
    np.testing.assert_allclose(right, robot_right)


def test_common_translation_moves_both_arms_and_keeps_width():
    controller = _controller()
    input_left, input_right, robot_left, robot_right, width = _lock(controller)
    displacement = np.array([0.12, -0.03, 0.08])
    input_left[:3, 3] += displacement
    input_right[:3, 3] += displacement

    left, right = controller.targets(input_left, input_right, now_s=0.1)

    np.testing.assert_allclose(left[:3, 3], robot_left[:3, 3] + displacement)
    np.testing.assert_allclose(right[:3, 3], robot_right[:3, 3] + displacement)
    assert np.linalg.norm(right[:3, 3] - left[:3, 3]) == pytest.approx(width)


def test_differential_translation_is_rejected():
    controller = _controller()
    input_left, input_right, robot_left, robot_right, _ = _lock(controller)
    input_left[0, 3] += 0.10
    input_right[0, 3] -= 0.10

    left, right = controller.targets(input_left, input_right, now_s=0.1)

    np.testing.assert_allclose(left, robot_left)
    np.testing.assert_allclose(right, robot_right)


def test_agreed_rotation_rotates_the_rigid_grasp_about_its_center():
    controller = _controller()
    input_left, input_right, robot_left, robot_right, width = _lock(controller)
    rotation = _rotation_z(np.deg2rad(90.0))
    input_left[:3, :3] = rotation
    input_right[:3, :3] = rotation

    left, right = controller.targets(input_left, input_right, now_s=0.1)
    center = 0.5 * (robot_left[:3, 3] + robot_right[:3, 3])

    np.testing.assert_allclose(
        left[:3, 3], center + rotation @ (robot_left[:3, 3] - center), atol=1e-7
    )
    np.testing.assert_allclose(
        right[:3, 3], center + rotation @ (robot_right[:3, 3] - center), atol=1e-7
    )
    np.testing.assert_allclose(left[:3, :3], rotation, atol=1e-12)
    np.testing.assert_allclose(right[:3, :3], rotation, atol=1e-12)
    assert np.linalg.norm(right[:3, 3] - left[:3, 3]) == pytest.approx(width)


def test_disagreed_rotation_is_rejected():
    controller = _controller()
    input_left, input_right, robot_left, robot_right, _ = _lock(controller)
    input_left[:3, :3] = _rotation_z(np.deg2rad(60.0))

    left, right = controller.targets(input_left, input_right, now_s=0.1)

    np.testing.assert_allclose(left, robot_left)
    np.testing.assert_allclose(right, robot_right)
    assert controller.last_rotation_disagreement_rad == pytest.approx(
        np.deg2rad(60.0)
    )


def test_rotation_recovery_rebases_without_catching_up():
    controller = _controller()
    input_left, input_right, robot_left, robot_right, _ = _lock(controller)
    input_left[:3, :3] = _rotation_z(np.deg2rad(60.0))
    left, right = controller.targets(input_left, input_right, now_s=0.1)
    np.testing.assert_allclose(left, robot_left)
    np.testing.assert_allclose(right, robot_right)
    assert controller.status()["rotation_suspended"]

    input_right[:3, :3] = _rotation_z(np.deg2rad(60.0))
    left, right = controller.targets(input_left, input_right, now_s=0.2)
    np.testing.assert_allclose(left, robot_left)
    np.testing.assert_allclose(right, robot_right)
    assert not controller.status()["rotation_suspended"]

    increment = _rotation_z(np.deg2rad(10.0))
    input_left[:3, :3] = increment @ input_left[:3, :3]
    input_right[:3, :3] = increment @ input_right[:3, :3]
    left, right = controller.targets(input_left, input_right, now_s=0.3)
    np.testing.assert_allclose(left[:3, :3], increment, atol=1e-12)
    np.testing.assert_allclose(right[:3, :3], increment, atol=1e-12)


def test_workspace_rejection_holds_last_valid_pair():
    controller = _controller()
    input_left, input_right, robot_left, robot_right, _ = _lock(controller)
    input_left[2, 3] += 0.20
    input_right[2, 3] += 0.20

    left, right = controller.targets(
        input_left,
        input_right,
        now_s=0.1,
        validator=lambda _left, _right: False,
    )

    np.testing.assert_allclose(left, robot_left)
    np.testing.assert_allclose(right, robot_right)
    assert controller.last_workspace_limited


def test_locked_motion_is_rate_limited_and_keeps_rigid_width():
    controller = DualObjectController(
        max_translation_speed_m_s=0.20,
        max_rotation_speed_rad_s=np.deg2rad(45.0),
    )
    input_left, input_right, robot_left, robot_right, width = _lock(controller)
    input_left[:3, 3] += [1.0, 0.0, 0.0]
    input_right[:3, 3] += [1.0, 0.0, 0.0]
    input_left[:3, :3] = _rotation_z(np.pi)
    input_right[:3, :3] = _rotation_z(np.pi)

    left, right = controller.targets(input_left, input_right, now_s=0.05)
    old_center = 0.5 * (robot_left[:3, 3] + robot_right[:3, 3])
    new_center = 0.5 * (left[:3, 3] + right[:3, 3])

    assert np.linalg.norm(new_center - old_center) == pytest.approx(0.01)
    assert np.linalg.norm(right[:3, 3] - left[:3, 3]) == pytest.approx(width)
    assert abs(np.arctan2(left[1, 0], left[0, 0])) == pytest.approx(
        np.deg2rad(2.25)
    )


def test_locked_motion_can_run_without_additional_speed_limits():
    controller = DualObjectController(
        max_translation_speed_m_s=None,
        max_rotation_speed_rad_s=None,
    )
    input_left, input_right, robot_left, robot_right, width = _lock(controller)
    displacement = np.array([0.50, -0.10, 0.20])
    rotation = _rotation_z(np.deg2rad(90.0))
    input_left[:3, 3] += displacement
    input_right[:3, 3] += displacement
    input_left[:3, :3] = rotation
    input_right[:3, :3] = rotation

    left, right = controller.targets(input_left, input_right, now_s=0.0)
    old_center = 0.5 * (robot_left[:3, 3] + robot_right[:3, 3])
    new_center = 0.5 * (left[:3, 3] + right[:3, 3])

    np.testing.assert_allclose(new_center, old_center + displacement)
    assert np.linalg.norm(right[:3, 3] - left[:3, 3]) == pytest.approx(width)
    np.testing.assert_allclose(left[:3, :3], rotation, atol=1e-12)


def test_unlock_rebases_independent_control_without_a_jump():
    controller = _controller()
    input_left, input_right, _, _, _ = _lock(controller)
    input_left[2, 3] += 0.10
    input_right[2, 3] += 0.10
    locked_left, locked_right = controller.targets(input_left, input_right, now_s=0.1)

    controller.unlock(input_left, input_right, locked_left, locked_right)
    left, right = controller.targets(input_left, input_right, now_s=0.2)
    np.testing.assert_allclose(left, locked_left)
    np.testing.assert_allclose(right, locked_right)
    assert controller.status()["lock_width_m"] is None
    assert controller.status()["rotation_disagreement_rad"] == 0.0
    assert not controller.status()["rotation_suspended"]

    input_left[0, 3] += 0.04
    left, right = controller.targets(input_left, input_right, now_s=0.3)
    np.testing.assert_allclose(left[:3, 3], locked_left[:3, 3] + [0.04, 0.0, 0.0])
    np.testing.assert_allclose(right, locked_right)


def test_rejects_invalid_pose():
    controller = _controller()
    with pytest.raises(ValueError, match="shape"):
        controller.lock(np.eye(3), np.eye(4), np.eye(4), np.eye(4), now_s=0.0)


class _FakeIK:
    def __init__(self):
        self.reset_values = []

    def reset_smoothing(self, value):
        self.reset_values.append(np.asarray(value).copy())


class _FakeArmController:
    def __init__(self):
        self.reset_values = []

    def reset_arm_command_reference(self, value):
        self.reset_values.append(np.asarray(value).copy())


class _TestDriver(RobotDriver):
    def _build_components(self):
        pass


def test_driver_toggle_uses_measured_fk_and_rebases_on_unlock(monkeypatch):
    driver = object.__new__(_TestDriver)
    driver._dual_object_control = _controller()
    driver._dual_object_toggle_requested = False
    driver._dual_object_last_warning_s = 0.0
    driver.simulation_mode = True
    driver.ik = _FakeIK()
    driver.controller = _FakeArmController()
    actual_left = _pose((0.3, 0.2, 1.0))
    actual_right = _pose((0.3, -0.2, 1.0))
    driver._get_current_ee_poses = lambda _q: (
        actual_left.copy(), actual_right.copy()
    )
    driver._dual_object_targets_valid = lambda _left, _right: True
    current_q = np.linspace(-0.3, 0.3, 14)
    current_dq = np.zeros(14)
    xr = XRProcessedData(
        left_wrist_pose=_pose((-0.25, 0.0, 0.0)),
        right_wrist_pose=_pose((0.25, 0.0, 0.0)),
        head_kwargs={},
    )

    assert driver.request_dual_object_toggle()
    locked_xr = driver._apply_dual_object_control(
        xr, current_q, current_dq, state_receive_ns=0, teleop_mode="relative_head"
    )
    assert driver.dual_object_locked
    np.testing.assert_allclose(locked_xr.left_wrist_pose, actual_left)
    np.testing.assert_allclose(locked_xr.right_wrist_pose, actual_right)
    assert len(driver.ik.reset_values) == 1
    assert len(driver.controller.reset_values) == 1

    assert driver.request_dual_object_toggle()
    unlocked_xr = driver._apply_dual_object_control(
        xr, current_q, current_dq, state_receive_ns=0, teleop_mode="relative_head"
    )
    assert not driver.dual_object_locked
    np.testing.assert_allclose(unlocked_xr.left_wrist_pose, actual_left)
    np.testing.assert_allclose(unlocked_xr.right_wrist_pose, actual_right)
    assert len(driver.ik.reset_values) == 2
    assert len(driver.controller.reset_values) == 2


def test_driver_toggle_is_supported_in_relative_pose_mode():
    driver = object.__new__(_TestDriver)
    driver._dual_object_control = _controller()
    driver._dual_object_toggle_requested = False
    driver._dual_object_last_warning_s = 0.0
    driver.simulation_mode = True
    driver.ik = _FakeIK()
    driver.controller = _FakeArmController()
    actual_left = _pose((0.3, 0.2, 1.0))
    actual_right = _pose((0.3, -0.2, 1.0))
    driver._get_current_ee_poses = lambda _q: (
        actual_left.copy(), actual_right.copy()
    )
    driver._dual_object_targets_valid = lambda _left, _right: True
    xr = XRProcessedData(
        left_wrist_pose=_pose((-0.25, 0.0, 0.0)),
        right_wrist_pose=_pose((0.25, 0.0, 0.0)),
        head_kwargs={},
    )

    assert driver.request_dual_object_toggle()
    output = driver._apply_dual_object_control(
        xr,
        np.zeros(14),
        np.zeros(14),
        state_receive_ns=0,
        teleop_mode="relative_pose",
    )

    assert driver.dual_object_locked
    np.testing.assert_allclose(output.left_wrist_pose, actual_left)
    np.testing.assert_allclose(output.right_wrist_pose, actual_right)


def test_driver_rejects_lock_while_arm_is_moving():
    driver = object.__new__(_TestDriver)
    driver._dual_object_control = _controller()
    driver._dual_object_toggle_requested = False
    driver._dual_object_last_warning_s = 0.0
    driver.simulation_mode = True
    driver.ik = _FakeIK()
    driver.controller = _FakeArmController()
    driver._get_current_ee_poses = lambda _q: (_pose(), _pose((0.0, 0.4, 0.0)))
    driver._dual_object_targets_valid = lambda _left, _right: True
    xr = XRProcessedData(_pose(), _pose((0.0, 0.4, 0.0)), {})

    driver.request_dual_object_toggle()
    driver._apply_dual_object_control(
        xr,
        np.zeros(14),
        np.full(14, 0.8),
        state_receive_ns=0,
        teleop_mode="relative_head",
    )

    assert not driver.dual_object_locked
    assert driver.ik.reset_values == []


def test_driver_rejects_toggle_when_fk_fails():
    driver = object.__new__(_TestDriver)
    driver._dual_object_control = _controller()
    driver._dual_object_toggle_requested = False
    driver._dual_object_last_warning_s = 0.0
    driver.simulation_mode = True
    driver.ik = _FakeIK()
    driver.controller = _FakeArmController()
    driver._dual_object_targets_valid = lambda _left, _right: True

    def fail_fk(_q):
        raise RuntimeError("bad FK")

    driver._get_current_ee_poses = fail_fk
    xr = XRProcessedData(_pose(), _pose((0.0, 0.4, 0.0)), {})

    driver.request_dual_object_toggle()
    output = driver._apply_dual_object_control(
        xr,
        np.zeros(14),
        np.zeros(14),
        state_receive_ns=0,
        teleop_mode="relative_head",
    )

    assert not driver.dual_object_locked
    np.testing.assert_allclose(output.left_wrist_pose, xr.left_wrist_pose)
    np.testing.assert_allclose(output.right_wrist_pose, xr.right_wrist_pose)
