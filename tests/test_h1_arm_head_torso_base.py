from types import SimpleNamespace

import numpy as np
import pytest

from teleop.teleop_h1_arm_head_torso_base import (
    H1BodyJoystickController,
    _prepare_argv,
)


class _StateBuffer:
    def __init__(self, state):
        self.state = state

    def get(self):
        return self.state


class _RosNode:
    def __init__(self, torso=(0.1, 0.2)):
        motors = [SimpleNamespace(q=float(value)) for value in torso]
        self.state_buffer = _StateBuffer(SimpleNamespace(motor_state=motors))
        self.base_commands = []

    def publish_base_cmd(self, vx, vy, wz):
        self.base_commands.append((vx, vy, wz))


class _ArmController:
    def __init__(self, torso=(0.1, 0.2)):
        self._ros_node = _RosNode(torso)
        self._raw_event_sink = None
        self.torso_commands = []

    def ctrl_torso(self, target):
        self.torso_commands.append(np.asarray(target, dtype=float))


def _tele(now_ns, *, left=(0.0, 0.0), right=(0.0, 0.0), left_click=False):
    return SimpleNamespace(
        controller_pose_timestamp_ns=now_ns,
        left_ctrl_thumbstickValue=np.asarray(left, dtype=float),
        right_ctrl_thumbstickValue=np.asarray(right, dtype=float),
        left_ctrl_thumbstick=left_click,
    )


def test_base_centers_to_zero_and_stale_input_hard_stops():
    body = H1BodyJoystickController()
    arm = _ArmController()
    now = 1_000_000_000

    moving = body.update_base(
        _tele(now, left=(0.0, -1.0)),
        arm,
        enabled=True,
        max_speed=1.0,
        frequency=20.0,
        now_ns=now,
    )
    assert moving[0] > 0.0

    centered = body.update_base(
        _tele(now + 50_000_000),
        arm,
        enabled=True,
        max_speed=1.0,
        frequency=20.0,
        now_ns=now + 50_000_000,
    )
    np.testing.assert_allclose(centered, 0.0)

    body.update_base(
        _tele(now + 100_000_000, left=(0.0, -1.0)),
        arm,
        enabled=True,
        max_speed=1.0,
        frequency=20.0,
        now_ns=now + 100_000_000,
    )
    stale = body.update_base(
        _tele(now, left=(0.0, -1.0)),
        arm,
        enabled=True,
        max_speed=1.0,
        frequency=20.0,
        now_ns=now + 500_000_000,
    )
    np.testing.assert_allclose(stale, 0.0)


def test_left_stick_click_changes_horizontal_axis_from_strafe_to_yaw():
    body = H1BodyJoystickController()
    arm = _ArmController()
    now = 2_000_000_000

    command = body.update_base(
        _tele(now, left=(-1.0, 0.0), left_click=True),
        arm,
        enabled=True,
        max_speed=1.0,
        frequency=20.0,
        now_ns=now,
    )
    assert command[1] == 0.0
    assert command[2] > 0.0


def test_torso_starts_from_lowstate_and_neutral_holds_target():
    body = H1BodyJoystickController()
    arm = _ArmController(torso=(0.1, 0.2))
    now = 3_000_000_000

    initial = body.update_torso(
        _tele(now), arm, enabled=True, frequency=20.0, now_ns=now
    )
    np.testing.assert_allclose(initial, [0.1, 0.2])

    moved = body.update_torso(
        _tele(now + 100_000_000, right=(1.0, -1.0)),
        arm,
        enabled=True,
        frequency=20.0,
        now_ns=now + 100_000_000,
    )
    np.testing.assert_allclose(moved, [0.105, 0.235], atol=1e-9)

    held = body.update_torso(
        _tele(now + 200_000_000),
        arm,
        enabled=True,
        frequency=20.0,
        now_ns=now + 200_000_000,
    )
    np.testing.assert_allclose(held, moved)
    assert body.recording_state(arm) == {
        "body_state": [0.1, 0.2],
        "body_action": pytest.approx([0.105, 0.235]),
    }


def test_entrypoint_adds_required_defaults_and_rejects_wrong_mode():
    argv = ["teleop_h1_arm_head_torso_base.py", "--record"]
    _prepare_argv(argv)
    assert argv[-7:] == [
        "--robot", "TOPSTAR_H1",
        "--input-mode", "controller",
        "--control-mode", "arms_head_torso",
        "--motion",
    ]

    with pytest.raises(SystemExit, match="requires --input-mode controller"):
        _prepare_argv(["script.py", "--input-mode", "hand"])

    with pytest.raises(SystemExit, match="requires --control-mode"):
        _prepare_argv(["script.py", "--control-mode", "arms_only"])
