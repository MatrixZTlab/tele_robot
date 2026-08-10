"""TOPSTAR H1 arm/head teleoperation with dedicated torso/base joysticks.

This entry point reuses ``teleop_hand_and_arm.py`` for arms, head, recording,
and XR button handling.  It only replaces the body joystick policy:

* Left stick Y/X: base forward/lateral velocity.
* Left stick click + X: base yaw velocity instead of lateral velocity.
* Right stick Y: torso pitch velocity.
* Right stick X: torso lift velocity (the H1 has no torso-yaw joint).

The hardware XAPI backend currently ignores ``/base_cmd``.  The base mapping is
therefore effective in MuJoCo/Isaac or with a real base driver that subscribes
to ``/base_cmd``; upper-body commands remain available on the real robot.
"""

from __future__ import annotations

import os
import runpy
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


PROFILE_ENV = "TELE_ROBOT_BODY_CONTROL_PROFILE"
PROFILE_NAME = "h1_arm_head_torso_base"


@dataclass(frozen=True)
class H1BodyControlConfig:
    deadzone: float = 0.08
    max_pico_age_s: float = 0.15
    base_linear_accel_m_s2: float = 0.8
    base_linear_decel_m_s2: float = 1.2
    base_angular_accel_rad_s2: float = 1.5
    base_angular_decel_rad_s2: float = 2.0
    torso_lift_speed_m_s: float = 0.05
    torso_pitch_speed_rad_s: float = 0.35
    torso_lift_limits_m: tuple[float, float] = (-0.01, 0.45)
    torso_pitch_limits_rad: tuple[float, float] = (0.0, 1.65806279)


class H1BodyJoystickController:
    """Stateful joystick mapper with stale-input and neutral-stop handling."""

    def __init__(self, config: H1BodyControlConfig | None = None):
        self.config = config or H1BodyControlConfig()
        self._base_current = np.zeros(3, dtype=float)
        self._base_last_ns = 0
        self._torso_target: np.ndarray | None = None
        self._torso_last_ns = 0

    def _axis(self, value: float) -> float:
        value = float(np.clip(value, -1.0, 1.0))
        magnitude = abs(value)
        if magnitude <= self.config.deadzone:
            return 0.0
        scaled = (magnitude - self.config.deadzone) / (1.0 - self.config.deadzone)
        smooth = scaled * scaled * (3.0 - 2.0 * scaled)
        return float(np.copysign(smooth, value))

    @staticmethod
    def _stick(tele_data: Any, side: str) -> np.ndarray:
        value = np.asarray(
            getattr(tele_data, f"{side}_ctrl_thumbstickValue", [0.0, 0.0]),
            dtype=float,
        ).reshape(-1)
        if value.size < 2 or not np.all(np.isfinite(value[:2])):
            return np.zeros(2, dtype=float)
        return np.clip(value[:2], -1.0, 1.0)

    def _pico_is_fresh(self, tele_data: Any, now_ns: int) -> bool:
        source_ns = int(getattr(tele_data, "controller_pose_timestamp_ns", 0) or 0)
        if source_ns <= 0:
            return False
        age_s = (now_ns - source_ns) / 1e9
        return -0.01 <= age_s <= self.config.max_pico_age_s

    @staticmethod
    def _dt(now_ns: int, previous_ns: int, frequency: float) -> float:
        nominal = 1.0 / max(float(frequency), 1.0)
        if previous_ns <= 0:
            return nominal
        return float(np.clip((now_ns - previous_ns) / 1e9, 0.001, 0.1))

    @staticmethod
    def _step(current: float, target: float, accel_step: float, decel_step: float) -> float:
        same_direction = (
            current == 0.0
            or target == 0.0
            or (current > 0.0) == (target > 0.0)
        )
        speeding_up = same_direction and abs(target) > abs(current)
        step = accel_step if speeding_up else decel_step
        return float(current + np.clip(target - current, -step, step))

    @staticmethod
    def _publish_base(arm_ctrl: Any, command: np.ndarray) -> None:
        ros_node = getattr(arm_ctrl, "_ros_node", None)
        publish = getattr(ros_node, "publish_base_cmd", None)
        if publish is not None:
            publish(float(command[0]), float(command[1]), float(command[2]))

    @staticmethod
    def _emit_raw(arm_ctrl: Any, stream: str, event: dict[str, Any]) -> None:
        sink = getattr(arm_ctrl, "_raw_event_sink", None)
        if sink is not None:
            sink(stream, event)

    def update_base(
        self,
        tele_data: Any,
        arm_ctrl: Any,
        *,
        enabled: bool,
        max_speed: float,
        frequency: float,
        now_ns: int | None = None,
    ) -> np.ndarray:
        now_ns = time.monotonic_ns() if now_ns is None else int(now_ns)
        fresh = self._pico_is_fresh(tele_data, now_ns)
        dt = self._dt(now_ns, self._base_last_ns, frequency)
        self._base_last_ns = now_ns

        target = np.zeros(3, dtype=float)
        if enabled and fresh:
            left = self._stick(tele_data, "left")
            forward = self._axis(-left[1])
            horizontal = self._axis(-left[0])
            speed = float(np.clip(abs(max_speed), 0.0, 1.0))
            target[0] = forward * speed
            if bool(getattr(tele_data, "left_ctrl_thumbstick", False)):
                target[2] = horizontal * speed
            else:
                target[1] = horizontal * speed

        if not enabled or not fresh:
            # Disarm and stale XR data are hard stops, not ramped stops.
            self._base_current[:] = 0.0
        else:
            cfg = self.config
            for index in (0, 1):
                self._base_current[index] = self._step(
                    self._base_current[index],
                    target[index],
                    cfg.base_linear_accel_m_s2 * dt,
                    cfg.base_linear_decel_m_s2 * dt,
                )
            self._base_current[2] = self._step(
                self._base_current[2],
                target[2],
                cfg.base_angular_accel_rad_s2 * dt,
                cfg.base_angular_decel_rad_s2 * dt,
            )

        command = self._base_current.copy()
        self._publish_base(arm_ctrl, command)
        self._emit_raw(arm_ctrl, "base_command", {
            "publish_monotonic_ns": now_ns,
            "publish_wall_ns": time.time_ns(),
            "pico_fresh": bool(fresh),
            "enabled": bool(enabled),
            "vx": float(command[0]),
            "vy": float(command[1]),
            "wz": float(command[2]),
        })
        return command

    @staticmethod
    def _read_torso_state(arm_ctrl: Any) -> np.ndarray | None:
        ros_node = getattr(arm_ctrl, "_ros_node", None)
        state_buffer = getattr(ros_node, "state_buffer", None)
        state = state_buffer.get() if state_buffer is not None else None
        if state is None or len(getattr(state, "motor_state", [])) < 2:
            return None
        value = np.array(
            [state.motor_state[0].q, state.motor_state[1].q], dtype=float
        )
        return value if np.all(np.isfinite(value)) else None

    def update_torso(
        self,
        tele_data: Any,
        arm_ctrl: Any,
        *,
        enabled: bool,
        frequency: float,
        now_ns: int | None = None,
    ) -> np.ndarray | None:
        now_ns = time.monotonic_ns() if now_ns is None else int(now_ns)
        measured = self._read_torso_state(arm_ctrl)
        if self._torso_target is None:
            if measured is None:
                return None
            self._torso_target = measured.copy()
            self._torso_last_ns = now_ns

        fresh = self._pico_is_fresh(tele_data, now_ns)
        dt = self._dt(now_ns, self._torso_last_ns, frequency)
        self._torso_last_ns = now_ns

        if enabled and fresh:
            right = self._stick(tele_data, "right")
            lift_rate = self._axis(right[0]) * self.config.torso_lift_speed_m_s
            pitch_rate = self._axis(-right[1]) * self.config.torso_pitch_speed_rad_s
            self._torso_target += np.array([lift_rate, pitch_rate]) * dt
            self._torso_target[0] = np.clip(
                self._torso_target[0], *self.config.torso_lift_limits_m
            )
            self._torso_target[1] = np.clip(
                self._torso_target[1], *self.config.torso_pitch_limits_rad
            )

        # Neutral or stale input holds the last target instead of returning home.
        arm_ctrl.ctrl_torso(self._torso_target.copy())
        self._emit_raw(arm_ctrl, "torso_command", {
            "publish_monotonic_ns": now_ns,
            "publish_wall_ns": time.time_ns(),
            "pico_fresh": bool(fresh),
            "enabled": bool(enabled),
            "q_target_hw": self._torso_target.tolist(),
        })
        return self._torso_target.copy()

    def recording_state(self, arm_ctrl: Any) -> dict[str, list[float]]:
        measured = self._read_torso_state(arm_ctrl)
        target = self._torso_target
        return {
            "body_state": measured.tolist() if measured is not None else [],
            "body_action": target.tolist() if target is not None else [],
        }

    def stop_base(self, arm_ctrl: Any) -> None:
        self._base_current[:] = 0.0
        self._publish_base(arm_ctrl, self._base_current)


def _option_value(argv: list[str], option: str) -> str | None:
    for index, value in enumerate(argv):
        if value == option and index + 1 < len(argv):
            return argv[index + 1]
        if value.startswith(option + "="):
            return value.split("=", 1)[1]
    return None


def _prepare_argv(argv: list[str]) -> None:
    robot = _option_value(argv, "--robot")
    if robot is not None and robot.upper() != "TOPSTAR_H1":
        raise SystemExit("This entry point only supports --robot TOPSTAR_H1")
    if robot is None:
        argv.extend(["--robot", "TOPSTAR_H1"])

    input_mode = _option_value(argv, "--input-mode")
    if input_mode is not None and input_mode != "controller":
        raise SystemExit("This entry point requires --input-mode controller")
    if input_mode is None:
        argv.extend(["--input-mode", "controller"])

    control_mode = _option_value(argv, "--control-mode")
    if control_mode is not None and control_mode != "arms_head_torso":
        raise SystemExit(
            "This entry point requires --control-mode arms_head_torso"
        )
    if control_mode is None:
        argv.extend(["--control-mode", "arms_head_torso"])

    if "--motion" not in argv:
        argv.append("--motion")


def main() -> None:
    _prepare_argv(sys.argv)
    os.environ[PROFILE_ENV] = PROFILE_NAME
    print(
        "H1 body controls: left stick=base XY, left-stick click+X=base yaw; "
        "right stick Y=torso pitch, X=torso lift",
        file=sys.stderr,
    )
    print(
        "WARNING: the current H1 XAPI backend ignores /base_cmd; real base motion "
        "requires a vendor base driver.",
        file=sys.stderr,
    )
    runpy.run_path(
        str(Path(__file__).with_name("teleop_hand_and_arm.py")),
        run_name="__main__",
    )


if __name__ == "__main__":
    main()
