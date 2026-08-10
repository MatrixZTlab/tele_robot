import threading
from types import SimpleNamespace

import numpy as np

from teleop.robot_control.topstar_h1.arm_controller import H1ArmController


def _motor(index):
    return SimpleNamespace(
        q=0.01 * index,
        dq=0.02 * index,
        tau_est=0.03 * index,
        temperature=[30 + index, 31 + index],
        motorstate=index,
        mode=1,
    )


def test_raw_lowstate_contains_force_and_fault_diagnostics():
    controller = object.__new__(H1ArmController)
    controller.left_slots = list(range(11, 18))
    controller.right_slots = list(range(4, 11))
    controller.joint_sign_map = np.array([
        1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    ])
    controller.publish_lock = threading.Lock()
    controller.last_published_q = np.linspace(-0.2, 0.2, 14)
    events = []
    controller._raw_event_sink = lambda stream, event: events.append((stream, event))
    msg = SimpleNamespace(motor_state=[_motor(index) for index in range(35)])

    controller._emit_lowstate_event(msg, {"sequence": 7})

    assert len(events) == 1
    stream, event = events[0]
    assert stream == "lowstate"
    assert event["sequence"] == 7
    assert len(event["q"]) == 14
    assert len(event["dq"]) == 14
    assert len(event["tau_est"]) == 14
    assert len(event["q_tracking_error"]) == 14
    assert len(event["temperature"]) == 14
    assert len(event["motorstate"]) == 14
    assert event["motorstate"][:2] == [11, 12]
    assert event["motorstate"][7:9] == [4, 5]


def test_fixed_gripper_disables_active_ee_command_recording():
    controller = object.__new__(H1ArmController)
    controller.publish_lock = threading.Lock()
    controller._ee_gripper_state = [0.0, 0.0]

    controller.disable_ee_gripper()

    assert controller._ee_gripper_state == []


def test_suction_cup_must_be_explicitly_enabled():
    controller = object.__new__(H1ArmController)
    controller.publish_lock = threading.Lock()
    controller._ee_gripper_state = []

    controller.enable_ee_gripper()

    assert controller._ee_gripper_state == [0.0, 0.0]
