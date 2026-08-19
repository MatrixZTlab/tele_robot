"""Isolated Pico-to-TOPSTAR_H1 live teleoperation using MuJoCo IK.

This entry point intentionally does not share the control loop in
``teleop_hand_and_arm.py``.  Dry-run is the default; robot commands are only
published when ``--live`` is explicitly supplied.
"""

from __future__ import annotations

import argparse
import logging
import math
import signal
import threading
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from teleop.robot_control._base.dual_object_control import DualObjectController
from teleop.robot_control.vr_mujoco_relative_teleop import (
    H1_XR_TO_ROBOT_ROTATION,
    H1MuJoCoLMIK,
    LMIKResult,
    VRRelativePoseTracker,
)
from teleop.robot_control.topstar_h1.joint_convention import (
    H1_ARM_COORDINATE_CONVENTION,
    H1_MODEL_REVISION,
)


LOG = logging.getLogger("h1_vr_mujoco_live")
H1_ARM_DOF = 14
CAMERA_SOURCES = ("head", "left_wrist", "right_wrist")
CAMERA_CONFIG_BY_SOURCE = {
    "head": "head_camera",
    "left_wrist": "left_wrist_camera",
    "right_wrist": "right_wrist_camera",
}
CAMERA_GETTER_BY_SOURCE = {
    "head": "get_head_frame",
    "left_wrist": "get_left_wrist_frame",
    "right_wrist": "get_right_wrist_frame",
}


def build_camera_key_plan(
    camera_config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Assign stable color/depth keys without importing the online LeRobot writer."""
    plan: list[dict[str, Any]] = []
    index = 0
    for source in CAMERA_SOURCES:
        camera_name = CAMERA_CONFIG_BY_SOURCE[source]
        config = (camera_config or {}).get(camera_name) or {}
        if not config.get("enable_zmq", False):
            continue
        halves = (0, 1) if config.get("binocular", False) else (None,)
        for half in halves:
            plan.append(
                {
                    "cam": camera_name,
                    "source": source,
                    "half": half,
                    "color_key": f"color_{index}",
                    "depth_key": (
                        f"depth_{index}"
                        if config.get("enable_depth", False)
                        else None
                    ),
                }
            )
            index += 1
    return plan
# A verified non-singular H1 arms-only pose from topstar_h1_test_004.  Dry-run
# uses this instead of all zeros so the viewer starts from a useful IK posture.
DEFAULT_DRY_RUN_INITIAL_Q = [
    -1.2305397,
    -1.1173229,
    -0.76580536,
    -0.7694363,
    0.54479694,
    0.348951,
    0.04157807,
    1.5395422,
    -1.3764126,
    0.25574026,
    -0.34151757,
    -0.25700444,
    0.34901664,
    -0.05125465,
]
H1_STATE_LIMIT_TOLERANCE_RAD = 0.03


class LiveSafetyError(RuntimeError):
    """A condition that requires teleoperation to be disarmed."""


@dataclass(frozen=True)
class LiveStepResult:
    command_q: np.ndarray
    raw_ik_q: np.ndarray
    ik: LMIKResult
    accepted_approximately: bool
    held_for_ik_safety: bool = False


@dataclass(frozen=True)
class CameraBatch:
    colors: dict[str, np.ndarray]
    depths: dict[str, np.ndarray]
    timestamps_ns: dict[str, int]


def build_lerobot_state_action(
    measured_q: Sequence[float], command_q: Sequence[float]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Match the existing H1 28-value LeRobot state/action schema."""
    state = _as_arm_q(measured_q, "measured_q")
    action = _as_arm_q(command_q, "command_q")
    states = {
        "left_arm": {"qpos": state[:7], "qvel": [], "torque": []},
        "right_arm": {"qpos": state[7:], "qvel": [], "torque": []},
        "left_ee": {"qpos": [], "qvel": [], "torque": []},
        "right_ee": {"qpos": [], "qvel": [], "torque": []},
        # Kept for compatibility with existing TOPSTAR_H1 recordings.
        "body": {"qpos": state},
    }
    actions = {
        "left_arm": {"qpos": action[:7], "qvel": [], "torque": []},
        "right_arm": {"qpos": action[7:], "qvel": [], "torque": []},
        "left_ee": {"qpos": [], "qvel": [], "torque": []},
        "right_ee": {"qpos": [], "qvel": [], "torque": []},
        "body": {"qpos": action},
    }
    return states, actions


def extract_camera_batch(
    camera_config: dict[str, Any],
    camera_key_plan: list[dict[str, Any]],
    images: dict[str, Any],
) -> CameraBatch:
    colors: dict[str, np.ndarray] = {}
    depths: dict[str, np.ndarray] = {}
    timestamps_ns: dict[str, int] = {}
    for entry in camera_key_plan:
        image = images.get(entry["source"])
        bgr = None if image is None else getattr(image, "bgr", None)
        if bgr is None:
            continue
        color = np.asarray(bgr)
        depth = getattr(image, "depth", None)
        half = entry["half"]
        if half is not None:
            width = int(camera_config[entry["cam"]]["image_shape"][1])
            half_width = width // 2
            columns = (
                slice(None, half_width)
                if half == 0
                else slice(half_width, None)
            )
            color = color[:, columns]
            if depth is not None:
                depth = np.asarray(depth)[:, columns]
        colors[entry["color_key"]] = np.ascontiguousarray(color)
        if entry["depth_key"] and depth is not None:
            depths[entry["depth_key"]] = np.ascontiguousarray(depth)
        timestamp_ns = getattr(image, "timestamp_ns", None)
        if timestamp_ns is not None:
            timestamps_ns[entry["color_key"]] = int(timestamp_ns)
    return CameraBatch(colors, depths, timestamps_ns)


class StateFreshnessMonitor:
    """Track whether a ROS state object has been replaced recently."""

    def __init__(self, timeout_s: float) -> None:
        if not math.isfinite(timeout_s) or timeout_s <= 0.0:
            raise ValueError("timeout_s must be finite and greater than zero")
        self.timeout_s = float(timeout_s)
        self._last_state: Any = None
        self._last_change_time: float | None = None

    def observe(self, state: Any, now: float | None = None) -> bool:
        timestamp = time.monotonic() if now is None else float(now)
        if state is None:
            return False
        if state is not self._last_state:
            self._last_state = state
            self._last_change_time = timestamp
        return (
            self._last_change_time is not None
            and timestamp - self._last_change_time <= self.timeout_s
        )


def _as_arm_q(value: Sequence[float], name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).reshape(-1)
    if result.shape != (H1_ARM_DOF,):
        raise LiveSafetyError(f"{name} must contain {H1_ARM_DOF} values")
    if not np.all(np.isfinite(result)):
        raise LiveSafetyError(f"{name} contains non-finite values")
    return result.copy()


def monotonic_timestamp_is_fresh(
    timestamp_ns: int,
    timeout_s: float,
    *,
    now_ns: int | None = None,
) -> bool:
    """Check an input timestamp from the local monotonic clock domain."""
    if not math.isfinite(timeout_s) or timeout_s <= 0.0:
        raise ValueError("timeout_s must be finite and greater than zero")
    try:
        source_ns = int(timestamp_ns)
    except (TypeError, ValueError, OverflowError):
        return False
    if source_ns <= 0:
        return False
    current_ns = time.monotonic_ns() if now_ns is None else int(now_ns)
    age_s = (current_ns - source_ns) / 1_000_000_000.0
    # A tiny negative age can occur when the producer samples just after the
    # caller. Larger future values indicate a clock-domain or data error.
    return -0.01 <= age_s <= timeout_s


def mapped_wrist_poses(
    tele_data: Any, controller_mapping: str
) -> tuple[np.ndarray, np.ndarray]:
    """Return controller poses ordered as robot-left, robot-right."""
    if controller_mapping == "same-side":
        return tele_data.left_wrist_pose, tele_data.right_wrist_pose
    if controller_mapping == "mirrored":
        return tele_data.right_wrist_pose, tele_data.left_wrist_pose
    raise ValueError(f"unknown controller mapping: {controller_mapping!r}")


def _read_robot_arm_state(
    controller: Any,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Read q, dq, and receipt time from one atomic LowState snapshot."""
    q, dq, receive_ns = controller.get_current_dual_arm_state()
    return (
        _as_arm_q(q, "measured arm q"),
        _as_arm_q(dq, "measured arm dq"),
        int(receive_ns),
    )


def _wait_for_fresh_robot_arm_state(
    controller: Any,
    *,
    freshness_timeout_s: float,
    wait_timeout_s: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Wait through a short post-MoveJ callback gap without accepting stale data."""
    deadline = time.monotonic() + float(wait_timeout_s)
    while time.monotonic() < deadline:
        q, dq, receive_ns = _read_robot_arm_state(controller)
        if monotonic_timestamp_is_fresh(receive_ns, freshness_timeout_s):
            return q, dq, receive_ns
        time.sleep(0.02)
    raise LiveSafetyError(
        f"/lowstate did not recover within {wait_timeout_s:.2f} seconds"
    )


def _validate_robot_arm_state(
    solver: H1MuJoCoLMIK,
    q: Sequence[float],
    dq: Sequence[float],
    *,
    limit_tolerance_rad: float = H1_STATE_LIMIT_TOLERANCE_RAD,
) -> tuple[np.ndarray, np.ndarray]:
    """Reject malformed measurements and states outside the H1 envelope."""
    measured_q = _as_arm_q(q, "measured arm q")
    measured_dq = _as_arm_q(dq, "measured arm dq")
    lower, upper = solver.arm_joint_limits
    within_limits = solver.arm_q_within_limits(
        measured_q, tolerance=limit_tolerance_rad
    )
    if within_limits:
        return measured_q, measured_dq

    distance = np.maximum(lower - measured_q, measured_q - upper)
    joint_index = int(np.argmax(distance))
    raise LiveSafetyError(
        f"measured arm q[{joint_index}]={measured_q[joint_index]:.4f} rad "
        f"is outside allowed [{lower[joint_index]:.4f}, "
        f"{upper[joint_index]:.4f}] rad envelope"
    )


def _send_robot_hold(
    controller: Any,
    solver: H1MuJoCoLMIK,
    measured_q: Sequence[float],
) -> np.ndarray:
    """Hold a validated measured pose with the proven gravity feed-forward."""
    command_q = _as_arm_q(measured_q, "hold arm q")
    tauff = _as_arm_q(
        solver.gravity_compensation(command_q),
        "gravity compensation",
    )
    controller.servo_dual_arm(command_q, tauff)
    return tauff


def _validate_ros_command_graph(
    controller: Any, *, settle_timeout_s: float = 2.0
) -> tuple[int, int]:
    """Require one local LowCmd publisher and at least one robot subscriber."""
    if not math.isfinite(settle_timeout_s) or settle_timeout_s < 0.0:
        raise ValueError("settle_timeout_s must be finite and non-negative")
    node = controller._ros_node
    deadline = time.monotonic() + settle_timeout_s
    publishers = subscribers = 0
    while True:
        publishers = int(node.count_publishers("/lowcmd"))
        subscribers = int(node.count_subscribers("/lowcmd"))
        if publishers == 1 and subscribers >= 1:
            return publishers, subscribers
        if time.monotonic() >= deadline:
            break
        time.sleep(0.05)
    if publishers != 1:
        raise LiveSafetyError(
            f"expected exactly one /lowcmd publisher (this process), found "
            f"{publishers}; stop every other arm controller"
        )
    raise LiveSafetyError(
        "no /lowcmd subscriber discovered; robot command endpoint is unavailable"
    )


def limit_joint_speed(
    previous_q: Sequence[float],
    target_q: Sequence[float],
    max_speed_rad_s: float,
    dt: float,
) -> np.ndarray:
    """Limit every joint independently without changing joint order."""
    previous = _as_arm_q(previous_q, "previous_q")
    target = _as_arm_q(target_q, "target_q")
    if not math.isfinite(max_speed_rad_s) or max_speed_rad_s <= 0.0:
        raise ValueError("max_speed_rad_s must be finite and greater than zero")
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and greater than zero")
    max_delta = max_speed_rad_s * dt
    return previous + np.clip(target - previous, -max_delta, max_delta)


class H1MuJoCoLiveCore:
    """Pure control core; ROS2 and TeleVuer are handled by the entry point."""

    def __init__(
        self,
        solver: H1MuJoCoLMIK,
        *,
        position_scale: float = 1.0,
        ema_alpha: float = 0.8,
        position_deadband_m: float = 0.01,
        rotation_deadband_deg: float = 3.0,
        orientation_control: str = "vr",
        max_joint_speed_rad_s: float = 0.5,
        max_solver_jump_rad: float = 0.35,
        max_tracking_error_rad: float = 0.35,
        tracking_error_check_enabled: bool = True,
        max_iters: int = 60,
        tolerance: float = 1e-4,
        damping: float = 1e-3,
        current_q_weight: float = 0.0001,
        home_qpos: Sequence[float] | None = None,
        home_weight: float = 0.0,
        max_ik_position_error_m: float = 0.06,
        max_ik_rotation_error_rad: float = math.radians(5.0),
        hold_on_ik_safety: bool = False,
        ik_safety_checks_enabled: bool = True,
        mirror_lateral_translation: bool = False,
    ) -> None:
        if not math.isfinite(max_solver_jump_rad) or max_solver_jump_rad <= 0.0:
            raise ValueError("max_solver_jump_rad must be finite and greater than zero")
        if (
            not math.isfinite(max_tracking_error_rad)
            or max_tracking_error_rad <= 0.0
        ):
            raise ValueError(
                "max_tracking_error_rad must be finite and greater than zero"
            )
        if orientation_control not in ("vr", "locked"):
            raise ValueError("orientation_control must be 'vr' or 'locked'")
        if (
            not math.isfinite(max_ik_position_error_m)
            or max_ik_position_error_m <= 0.0
        ):
            raise ValueError(
                "max_ik_position_error_m must be finite and greater than zero"
            )
        if (
            not math.isfinite(max_ik_rotation_error_rad)
            or max_ik_rotation_error_rad <= 0.0
        ):
            raise ValueError(
                "max_ik_rotation_error_rad must be finite and greater than zero"
            )
        self.solver = solver
        self.position_scale = position_scale
        self.ema_alpha = ema_alpha
        self.position_deadband_m = position_deadband_m
        self.rotation_deadband_deg = rotation_deadband_deg
        self.orientation_control = orientation_control
        self.max_joint_speed_rad_s = max_joint_speed_rad_s
        self.max_solver_jump_rad = max_solver_jump_rad
        self.max_tracking_error_rad = max_tracking_error_rad
        self.tracking_error_check_enabled = bool(tracking_error_check_enabled)
        self.max_iters = max_iters
        self.tolerance = tolerance
        self.damping = damping
        self.current_q_weight = current_q_weight
        if home_qpos is None:
            self.home_qpos: np.ndarray | None = None
        else:
            self.home_qpos = _as_arm_q(home_qpos, "home_qpos")
        if not math.isfinite(home_weight) or home_weight < 0.0:
            raise ValueError("home_weight must be finite and non-negative")
        self.home_weight = float(home_weight)
        self.max_ik_position_error_m = max_ik_position_error_m
        self.max_ik_rotation_error_rad = max_ik_rotation_error_rad
        self.hold_on_ik_safety = bool(hold_on_ik_safety)
        self.ik_safety_checks_enabled = bool(ik_safety_checks_enabled)
        self.mirror_lateral_translation = bool(mirror_lateral_translation)
        self.left_tracker: VRRelativePoseTracker | None = None
        self.right_tracker: VRRelativePoseTracker | None = None
        self.previous_command_q: np.ndarray | None = None
        self.previous_raw_ik_q: np.ndarray | None = None
        # Reuse the proven rigid-grasp controller from teleop_hand_and_arm.py.
        # It operates on the already mapped/scaled robot EE targets here, so
        # mirrored controller mapping and position scale remain unchanged.
        self.dual_object_controller = DualObjectController(
            max_translation_speed_m_s=None,
            max_rotation_speed_rad_s=None,
            max_rotation_disagreement_rad=math.radians(25.0),
        )
        self.active = False

    @property
    def dual_object_locked(self) -> bool:
        return self.dual_object_controller.locked

    def arm(
        self,
        current_q: Sequence[float],
        left_wrist_pose: np.ndarray,
        right_wrist_pose: np.ndarray,
    ) -> None:
        current = _as_arm_q(current_q, "current_q")
        left_ee, right_ee = self.solver.forward_kinematics(current)
        tracker_kwargs = dict(
            position_scale=self.position_scale,
            ema_alpha=self.ema_alpha,
            position_deadband=self.position_deadband_m,
            rotation_deadband_deg=self.rotation_deadband_deg,
            lock_orientation=self.orientation_control == "locked",
            translation_axis_sign=(
                (1.0, -1.0, 1.0)
                if self.mirror_lateral_translation
                else (1.0, 1.0, 1.0)
            ),
        )
        self.left_tracker = VRRelativePoseTracker(
            left_ee, H1_XR_TO_ROBOT_ROTATION, **tracker_kwargs
        )
        self.right_tracker = VRRelativePoseTracker(
            right_ee, H1_XR_TO_ROBOT_ROTATION, **tracker_kwargs
        )
        self.left_tracker.update(left_wrist_pose)
        self.right_tracker.update(right_wrist_pose)
        self.previous_command_q = current
        self.previous_raw_ik_q = current.copy()
        self.dual_object_controller.reset()
        self.active = True

    def toggle_dual_object_lock(
        self,
        current_q: Sequence[float],
        left_wrist_pose: np.ndarray,
        right_wrist_pose: np.ndarray,
        *,
        now_s: float | None = None,
    ) -> tuple[bool, float | None]:
        """Lock/unlock the measured dual-arm grasp without a target jump."""
        if not self.active or self.left_tracker is None or self.right_tracker is None:
            raise LiveSafetyError("teleoperation must be armed before box lock")

        current = _as_arm_q(current_q, "current_q")
        left_input = self.left_tracker.update(left_wrist_pose)
        right_input = self.right_tracker.update(right_wrist_pose)
        actual_left, actual_right = self.solver.forward_kinematics(current)
        width_m = float(
            np.linalg.norm(actual_right[:3, 3] - actual_left[:3, 3])
        )
        if not 0.10 <= width_m <= 1.50:
            raise LiveSafetyError(
                f"measured grasp width {width_m:.3f} m is outside [0.10, 1.50] m"
            )

        if self.dual_object_controller.locked:
            self.dual_object_controller.unlock(
                left_input,
                right_input,
                actual_left,
                actual_right,
            )
            locked = False
            lock_width = None
        else:
            lock_width = self.dual_object_controller.lock(
                left_input,
                right_input,
                actual_left,
                actual_right,
                now_s=time.monotonic() if now_s is None else float(now_s),
            )
            locked = True

        # Rebase the command-side safety references to the same measured pose.
        # This prevents a legitimate lock transition from looking like an IK
        # branch jump or tracking error on its first frame.
        self.previous_command_q = current.copy()
        self.previous_raw_ik_q = current.copy()
        return locked, lock_width

    def disarm(self) -> None:
        self.active = False
        self.left_tracker = None
        self.right_tracker = None
        self.previous_command_q = None
        self.previous_raw_ik_q = None
        self.dual_object_controller.reset()

    def _hold_previous_command(self, result: LMIKResult) -> LiveStepResult:
        if self.previous_command_q is None or self.previous_raw_ik_q is None:
            raise LiveSafetyError("teleoperation state is incomplete")
        return LiveStepResult(
            self.previous_command_q.copy(),
            self.previous_raw_ik_q.copy(),
            result,
            False,
            held_for_ik_safety=True,
        )

    def step(
        self,
        current_q: Sequence[float],
        left_wrist_pose: np.ndarray,
        right_wrist_pose: np.ndarray,
        dt: float,
    ) -> LiveStepResult:
        if not self.active:
            raise LiveSafetyError("teleoperation is not armed")
        if (
            self.left_tracker is None
            or self.right_tracker is None
            or self.previous_command_q is None
            or self.previous_raw_ik_q is None
        ):
            raise LiveSafetyError("teleoperation state is incomplete")

        current = _as_arm_q(current_q, "current_q")
        tracking_error = float(
            np.max(np.abs(self.previous_command_q - current))
        )
        if (
            self.tracking_error_check_enabled
            and tracking_error > self.max_tracking_error_rad
        ):
            raise LiveSafetyError(
                f"robot tracking error {tracking_error:.4f} rad exceeds "
                f"{self.max_tracking_error_rad:.4f} rad"
            )
        left_target = self.left_tracker.update(left_wrist_pose)
        right_target = self.right_tracker.update(right_wrist_pose)
        left_target, right_target = self.dual_object_controller.targets(
            left_target,
            right_target,
            now_s=time.monotonic(),
        )
        result = self.solver.solve(
            left_target,
            right_target,
            current,
            max_iters=self.max_iters,
            tolerance=self.tolerance,
            damping=self.damping,
            current_q_weight=self.current_q_weight,
            home_qpos=self.home_qpos,
            home_weight=self.home_weight,
        )
        accepted_approximately = False
        if not result.converged:
            if self.ik_safety_checks_enabled and (
                result.translation_error_norm > self.max_ik_position_error_m
                or result.rotation_error_norm > self.max_ik_rotation_error_rad
            ):
                if self.hold_on_ik_safety:
                    return self._hold_previous_command(result)
                raise LiveSafetyError(
                    "MuJoCo IK did not converge within the live safety envelope "
                    f"(position={result.translation_error_norm:.4f} m, "
                    f"rotation={result.rotation_error_norm:.4f} rad)"
                )
            accepted_approximately = True

        raw_ik_q = _as_arm_q(result.arm_q, "IK result")
        if not self.solver.arm_q_within_limits(raw_ik_q, tolerance=1e-9):
            raise LiveSafetyError("MuJoCo IK returned a joint-limit violation")
        # Compare consecutive IK solutions, not the speed-limited command. A
        # user can legitimately move faster than the configured robot speed,
        # in which case the command lags the target by design.
        solver_jump = float(np.max(np.abs(raw_ik_q - self.previous_raw_ik_q)))
        if (
            self.ik_safety_checks_enabled
            and solver_jump > self.max_solver_jump_rad
        ):
            if self.hold_on_ik_safety:
                return self._hold_previous_command(result)
            raise LiveSafetyError(
                f"IK joint jump {solver_jump:.4f} rad exceeds "
                f"{self.max_solver_jump_rad:.4f} rad"
            )

        command_q = limit_joint_speed(
            self.previous_command_q,
            raw_ik_q,
            self.max_joint_speed_rad_s,
            dt,
        )
        command_within_limits = self.solver.arm_q_within_limits(
            command_q, tolerance=H1_STATE_LIMIT_TOLERANCE_RAD
        )
        if not command_within_limits:
            raise LiveSafetyError(
                "speed-limited command is outside the H1 joint envelope"
            )
        self.previous_command_q = command_q.copy()
        self.previous_raw_ik_q = raw_ik_q.copy()
        return LiveStepResult(
            command_q, raw_ik_q, result, accepted_approximately
        )


class FormalRecordingSession:
    """Record raw sensor/control streams without duplicate trajectory files.

    By default the live entry point does not provide ``legacy_episode_writer``:
    compressed camera packets are written once under ``raw/episode_XXXX`` and
    image alignment/legacy/LeRobot conversion are offline steps.  Supplying a
    writer is retained for compatibility and tests, but must not be used on the
    latency-sensitive raw capture path.
    """

    def __init__(
        self,
        *,
        image_client: Any,
        camera_config: dict[str, Any],
        camera_key_plan: list[dict[str, Any]],
        legacy_episode_writer: Any | None,
        output_root: Path,
        task_name: str,
        task_description: str,
        frequency: float,
        sync_tolerance_s: float,
        camera_sync_tolerance_s: float,
        raw_writer: Any | None = None,
        max_camera_age_s: float = 0.10,
    ) -> None:
        self.image_client = image_client
        self.camera_config = camera_config
        self.camera_key_plan = camera_key_plan
        self.enabled_camera_sources = tuple(
            dict.fromkeys(entry["source"] for entry in camera_key_plan)
        )
        if not self.enabled_camera_sources:
            raise ValueError("recording requires at least one enabled camera")
        self.legacy_episode_writer = legacy_episode_writer
        self.output_root = output_root
        self.task_name = task_name
        self.task_description = task_description
        self.frequency = frequency
        self.sync_tolerance_s = sync_tolerance_s
        self.camera_sync_tolerance_s = camera_sync_tolerance_s
        self.raw_writer = raw_writer
        if self.raw_writer is None and self.legacy_episode_writer is None:
            raise ValueError("recording requires a raw or legacy writer")
        self.max_camera_age_s = float(max_camera_age_s)
        self.active = False
        self.episode_index = self._next_episode_index()
        self.last_saved_episode_index: int | None = None
        self.frame_index = 0
        self._prefetched_images: dict[str, Any] | None = None
        self._raw_listeners: list[tuple[str, Any]] = []
        if self.raw_writer is not None and hasattr(
            self.image_client, "add_packet_listener"
        ):
            for source in self.enabled_camera_sources:
                camera_name = CAMERA_CONFIG_BY_SOURCE[source]
                listener = self.raw_writer.append_camera_packet
                try:
                    self.image_client.add_packet_listener(camera_name, listener)
                    self._raw_listeners.append((camera_name, listener))
                except Exception:
                    LOG.exception("failed to attach raw listener: %s", camera_name)

    def _next_episode_index(self) -> int:
        """Choose an unused raw episode id across repeated program launches."""
        if self.legacy_episode_writer is not None:
            return 0
        raw_root = self.output_root / "raw"
        indices = []
        if raw_root.exists():
            for path in raw_root.glob("episode_*"):
                try:
                    indices.append(int(path.name.rsplit("_", 1)[1]))
                except (IndexError, ValueError):
                    continue
        return max(indices, default=-1) + 1

    def _read_images(self) -> dict[str, Any]:
        return {
            source: getattr(
                self.image_client, CAMERA_GETTER_BY_SOURCE[source]
            )()
            for source in self.enabled_camera_sources
        }

    def _missing_physical_cameras(self, images: dict[str, Any]) -> list[str]:
        missing = []
        for source in self.enabled_camera_sources:
            image = images.get(source)
            if image is None:
                missing.append(source)
                continue
            if self.legacy_episode_writer is not None:
                available = getattr(image, "bgr", None) is not None
            else:
                available = bool(getattr(image, "jpg", None))
            if not available:
                missing.append(source)
        return missing

    def start(self) -> None:
        if self.active:
            return
        images = self._read_images()
        missing = self._missing_physical_cameras(images)
        if missing:
            raise RuntimeError(
                "recording not started; missing camera frames: "
                + ", ".join(missing)
            )

        try:
            if self.legacy_episode_writer is not None:
                if not self.legacy_episode_writer.create_episode():
                    raise RuntimeError("legacy episode writer is busy")
                self.episode_index = int(
                    getattr(
                        self.legacy_episode_writer,
                        "episode_id",
                        self.episode_index,
                    )
                )
            if self.raw_writer is not None:
                self.raw_writer.start_episode(
                    self.episode_index,
                    metadata={
                        "task_name": self.task_name,
                        "task_description": self.task_description,
                        "frequency_hz": self.frequency,
                        "control_mode": "incremental_mujoco_ik",
                        "robot_model": "TOPSTAR_H1_MUJOCO_IK",
                        "robot_model_revision": H1_MODEL_REVISION,
                        "arm_coordinate_convention": (
                            H1_ARM_COORDINATE_CONVENTION
                        ),
                    },
                )
        except Exception:
            if self.legacy_episode_writer is not None:
                try:
                    self.legacy_episode_writer.discard_episode(self.episode_index)
                except Exception:
                    pass
            raise

        self._prefetched_images = images
        self.frame_index = 0
        self.active = True

    def capture_camera_batch(self) -> CameraBatch:
        if not self.active:
            raise RuntimeError("recording is not active")
        images = self._prefetched_images or self._read_images()
        self._prefetched_images = None
        if self.legacy_episode_writer is not None:
            return extract_camera_batch(
                self.camera_config, self.camera_key_plan, images
            )
        timestamps_ns: dict[str, int] = {}
        for entry in self.camera_key_plan:
            image = images.get(entry["source"])
            timestamp_ns = (
                None if image is None else getattr(image, "timestamp_ns", None)
            )
            if timestamp_ns is not None:
                timestamps_ns[entry["color_key"]] = int(timestamp_ns)
        return CameraBatch({}, {}, timestamps_ns)

    def add_frame(
        self,
        *,
        camera_batch: CameraBatch,
        measured_q: Sequence[float],
        command_q: Sequence[float],
        raw_ik_q: Sequence[float],
        ik_result: LMIKResult,
        accepted_approximately: bool,
        control_cycle_timestamp_ns: int,
        robot_step_done_timestamp_ns: int,
        pico_timestamp_ns: int | None = None,
        state_receive_timestamp_ns: int | None = None,
    ) -> None:
        if not self.active:
            raise RuntimeError("recording is not active")
        camera_timestamps = list(camera_batch.timestamps_ns.values())
        camera_sync_skew_s = None
        if camera_timestamps:
            camera_sync_skew_s = (
                max(camera_timestamps) - min(camera_timestamps)
            ) / 1_000_000_000.0
        camera_ages_s = [
            max(
                0.0,
                (int(control_cycle_timestamp_ns) - int(timestamp))
                / 1_000_000_000.0,
            )
            for timestamp in camera_timestamps
        ]
        max_camera_age_s = max(camera_ages_s) if camera_ages_s else None
        invalid_reasons = []
        if (
            camera_sync_skew_s is not None
            and camera_sync_skew_s > self.camera_sync_tolerance_s
        ):
            invalid_reasons.append("camera_sync_skew")
        if (
            max_camera_age_s is not None
            and max_camera_age_s > self.max_camera_age_s
        ):
            invalid_reasons.append("camera_age")
        alignment = {
            "scheme": "lerobot_fps_grid",
            "fps": self.frequency,
            "tolerance_s": self.sync_tolerance_s,
            "camera_sync_tolerance_s": self.camera_sync_tolerance_s,
            "actual_timestamps_ns": {
                "control_cycle": int(control_cycle_timestamp_ns),
                "robot_step_done": int(robot_step_done_timestamp_ns),
                "cameras": camera_batch.timestamps_ns,
            },
            "camera_sync_skew_s": camera_sync_skew_s,
            "camera_sync_within_tolerance": (
                None
                if camera_sync_skew_s is None
                else camera_sync_skew_s <= self.camera_sync_tolerance_s
            ),
            "max_camera_age_s": max_camera_age_s,
            "camera_age_within_tolerance": (
                None
                if max_camera_age_s is None
                else max_camera_age_s <= self.max_camera_age_s
            ),
            "valid": not invalid_reasons,
            "invalid_reasons": invalid_reasons,
            "state_receive_timestamp_ns": state_receive_timestamp_ns,
            "pico_timestamp_ns": pico_timestamp_ns,
            "mujoco_ik": {
                "raw_arm_q": _as_arm_q(raw_ik_q, "raw_ik_q").tolist(),
                "converged": bool(ik_result.converged),
                "accepted_approximately": bool(accepted_approximately),
                "iterations": int(ik_result.iterations),
                "translation_error_norm": float(
                    ik_result.translation_error_norm
                ),
                "rotation_error_norm": float(ik_result.rotation_error_norm),
            },
        }
        if self.legacy_episode_writer is not None:
            states, actions = build_lerobot_state_action(measured_q, command_q)

            def _json_tree(value: Any) -> Any:
                if isinstance(value, np.ndarray):
                    return value.tolist()
                if isinstance(value, dict):
                    return {key: _json_tree(item) for key, item in value.items()}
                if isinstance(value, (list, tuple)):
                    return [_json_tree(item) for item in value]
                return value

            legacy_alignment = dict(alignment)
            legacy_alignment["max_camera_age_s"] = self.max_camera_age_s
            self.legacy_episode_writer.add_item(
                colors=camera_batch.colors,
                depths=camera_batch.depths,
                states=_json_tree(states),
                actions=_json_tree(actions),
                alignment=_json_tree(legacy_alignment),
            )

        self.frame_index += 1

        if self.raw_writer is not None:
            self.raw_writer.append_event(
                "control",
                {
                    "frame_index": int(self.frame_index - 1),
                    "control_cycle_timestamp_ns": int(control_cycle_timestamp_ns),
                    "robot_step_done_timestamp_ns": int(robot_step_done_timestamp_ns),
                    "pico_timestamp_ns": pico_timestamp_ns,
                    "state_receive_timestamp_ns": state_receive_timestamp_ns,
                    "camera_timestamps_ns": {
                        key: int(value)
                        for key, value in camera_batch.timestamps_ns.items()
                    },
                    "measured_q_rad": _as_arm_q(measured_q, "measured_q").tolist(),
                    "command_q_rad": _as_arm_q(command_q, "command_q").tolist(),
                    "raw_ik_q_rad": _as_arm_q(raw_ik_q, "raw_ik_q").tolist(),
                    "ik_converged": bool(ik_result.converged),
                    "ik_translation_error_m": float(ik_result.translation_error_norm),
                    "ik_rotation_error_rad": float(ik_result.rotation_error_norm),
                },
            )

    def stop(self) -> None:
        if not self.active:
            return
        errors = []
        try:
            if self.raw_writer is not None:
                self.raw_writer.stop_episode()
        except Exception as exc:
            errors.append(f"raw stream save failed: {exc}")
        if self.legacy_episode_writer is not None:
            try:
                self.legacy_episode_writer.save_episode()
            except Exception as exc:
                errors.append(f"legacy episode save failed: {exc}")
        self.active = False
        self._prefetched_images = None
        self.last_saved_episode_index = int(self.episode_index)
        self.episode_index += 1
        if errors:
            raise RuntimeError("; ".join(errors))

    def discard(self) -> None:
        """Discard the current episode using the old Right-B semantics."""
        if not self.active:
            return
        episode_index = int(self.episode_index)
        errors = []
        if self.raw_writer is not None:
            try:
                self.raw_writer.discard_episode(episode_index)
            except Exception as exc:
                errors.append(f"raw discard failed: {exc}")
        if self.legacy_episode_writer is not None:
            try:
                self.legacy_episode_writer.discard_episode(episode_index)
            except Exception as exc:
                errors.append(f"legacy discard failed: {exc}")
        self.active = False
        self._prefetched_images = None
        self.episode_index += 1
        if errors:
            raise RuntimeError("; ".join(errors))

    def discard_latest(self) -> None:
        """Discard the most recently saved old-format episode, if present."""
        episode_index = self.last_saved_episode_index
        if episode_index is None:
            return
        errors = []
        if self.raw_writer is not None:
            try:
                self.raw_writer.discard_episode(episode_index)
            except Exception as exc:
                errors.append(f"raw discard failed: {exc}")
        if self.legacy_episode_writer is not None:
            try:
                self.legacy_episode_writer.discard_episode(episode_index)
            except Exception as exc:
                errors.append(f"legacy discard failed: {exc}")
        self.last_saved_episode_index = None
        if errors:
            raise RuntimeError("; ".join(errors))

    def close(self) -> None:
        stop_error = None
        try:
            self.stop()
        except Exception as exc:
            stop_error = exc
        try:
            if self.legacy_episode_writer is not None:
                self.legacy_episode_writer.close()
        finally:
            if self.raw_writer is not None:
                self.raw_writer.close()
        if stop_error is not None:
            raise stop_error


class H1MuJoCoViewer:
    """Optional native MuJoCo viewer for the same model used by live IK."""

    def __init__(self, solver: H1MuJoCoLMIK) -> None:
        try:
            import mujoco.viewer
        except ImportError as exc:
            raise RuntimeError(
                "MuJoCo viewer requires the optional GUI dependencies"
            ) from exc
        self._solver = solver
        threads_before_launch = set(threading.enumerate())
        self._viewer = mujoco.viewer.launch_passive(
            solver._model,
            solver._data,
            show_left_ui=True,
            show_right_ui=True,
        )
        self._viewer_threads = [
            thread
            for thread in threading.enumerate()
            if thread not in threads_before_launch
        ]

    def is_running(self) -> bool:
        return self._viewer.is_running()

    def update(self, arm_q: Sequence[float]) -> None:
        with self._viewer.lock():
            self._solver.forward_kinematics(
                _as_arm_q(arm_q, "viewer arm_q")
            )
        self._viewer.sync()

    def lock(self) -> Any:
        """Lock the passive viewer while its shared MuJoCo data is changed."""
        return self._viewer.lock()

    def close(self) -> None:
        self._viewer.close()
        deadline = time.monotonic() + 2.0
        for thread in self._viewer_threads:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                break
            thread.join(remaining)
        still_running = [
            thread.name for thread in self._viewer_threads if thread.is_alive()
        ]
        if still_running:
            LOG.warning(
                "MuJoCo viewer threads did not stop before timeout: %s",
                ", ".join(still_running),
            )


def _build_parser() -> argparse.ArgumentParser:
    default_task_dir = Path(__file__).resolve().parent / "utils" / "data"
    parser = argparse.ArgumentParser(
        description="Isolated TOPSTAR_H1 Pico teleoperation with MuJoCo IK"
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="publish commands to the real H1; omitted means dry-run",
    )
    parser.add_argument(
        "--allow-hard-limit-start",
        action="store_true",
        help=(
            "use mechanical hard limits for live startup and IK so enabling "
            "holds the measured pose even inside the normal 5-degree margin"
        ),
    )
    parser.add_argument("--frequency", type=float, default=20.0)
    parser.add_argument(
        "--arm-scale",
        type=float,
        default=0.7,
        help="VR translation scale; 0.7 is validated against recorded H1 motion",
    )
    parser.add_argument(
        "--controller-mapping",
        choices=("same-side", "mirrored"),
        default=None,
        help=(
            "same-side maps Pico left/right to robot left/right; mirrored maps "
            "Pico left to robot right and Pico right to robot left, matching "
            "face-to-face operation and reverses lateral translation. Required "
            "with --live"
        ),
    )
    parser.add_argument("--ema-alpha", type=float, default=0.8)
    parser.add_argument("--position-deadband-m", type=float, default=0.01)
    parser.add_argument("--rotation-deadband-deg", type=float, default=3.0)
    parser.add_argument(
        "--orientation-control",
        choices=("vr", "locked"),
        default="vr",
        help=(
            "vr maps controller wrist rotation to each end effector; locked "
            "captures each measured end-effector rotation when Left A arms "
            "teleoperation and then accepts translation only"
        ),
    )
    parser.add_argument("--max-joint-speed", type=float, default=0.5)
    parser.add_argument(
        "--max-solver-jump",
        type=float,
        default=None,
        help=(
            "maximum raw IK joint branch jump in rad; defaults to 0.8 in "
            "dry-run and 0.35 with --live"
        ),
    )
    parser.add_argument("--max-tracking-error", type=float, default=0.35)
    parser.add_argument(
        "--disable-tracking-error-check",
        action="store_true",
        help=(
            "do not disarm when commanded and measured arm joints diverge; "
            "mechanical limits, IK jump checks, ROS graph checks, and state "
            "timeouts remain active"
        ),
    )
    parser.add_argument("--state-timeout", type=float, default=0.25)
    parser.add_argument(
        "--legacy-stale-state-behavior",
        action="store_true",
        help=(
            "match the old teleop loop during brief /lowstate gaps by using "
            "the last valid measured arm state instead of disarming; a hard "
            "timeout still stops live control"
        ),
    )
    parser.add_argument(
        "--legacy-stale-state-hard-timeout",
        type=float,
        default=2.0,
        help=(
            "maximum /lowstate age allowed by legacy compatibility mode "
            "before live control is disarmed"
        ),
    )
    parser.add_argument(
        "--pico-timeout",
        type=float,
        default=0.25,
        help="maximum age in seconds of the latest Pico controller frame",
    )
    parser.add_argument(
        "--max-arm-speed-at-arm",
        type=float,
        default=0.2,
        help="maximum measured absolute arm velocity allowed when arming (rad/s)",
    )
    parser.add_argument("--ik-max-iters", type=int, default=60)
    parser.add_argument(
        "--ik-tolerance",
        type=float,
        default=1e-4,
        help="high-precision solver residual threshold; lower values produce motion",
    )
    parser.add_argument(
        "--max-ik-position-error",
        type=float,
        default=0.06,
        help=(
            "IK position-error safety envelope (m); above this the frame is "
            "held in dry-run or disarmed in live mode"
        ),
    )
    parser.add_argument("--max-ik-rotation-error-deg", type=float, default=5.0)
    parser.add_argument(
        "--disable-ik-safety-checks",
        action="store_true",
        help=(
            "dry-run only: accept finite in-limit IK output even when the "
            "solver does not converge or changes branches"
        ),
    )
    parser.add_argument("--ik-damping", type=float, default=1e-3)
    parser.add_argument(
        "--current-q-weight",
        type=float,
        default=0.0001,
        help=(
            "small joint-space pullback toward the solve's starting state; "
            "0.0001 prevented redundant-arm branch flips in recorded H1 replay "
            "without the large task residual caused by the previous 0.01 default"
        ),
    )
    parser.add_argument(
        "--home-weight",
        type=float,
        default=0.0,
        help=(
            "optional extra pull toward a fixed natural arm posture, on top of "
            "--current-q-weight. Off by default; raise slightly (e.g. 0.01) only "
            "if the arms still drift toward awkward configurations"
        ),
    )
    parser.add_argument(
        "--mujoco-viewer",
        action="store_true",
        help="show the H1 MuJoCo window with the current arm command",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help=(
            "record compressed raw camera, LowState, LowCmd, Pico, and control "
            "streams; generate aligned legacy/LeRobot data offline"
        ),
    )
    parser.add_argument(
        "--record-cameras",
        nargs="+",
        choices=CAMERA_SOURCES,
        default=None,
        help=(
            "physical cameras to record; omitted requires head, left_wrist, "
            "and right_wrist, while e.g. '--record-cameras head' explicitly "
            "enables head-only recording"
        ),
    )
    parser.add_argument(
        "--no-rerun",
        action="store_true",
        help="disable the old EpisodeWriter Rerun visualization",
    )
    parser.add_argument(
        "--no-raw-record",
        dest="raw_record",
        action="store_false",
        help="disable asynchronous raw camera/Pico/ROS stream recording",
    )
    parser.set_defaults(raw_record=True)
    parser.add_argument(
        "--raw-record-queue-size",
        type=int,
        default=8192,
        help="maximum queued raw events before drops are counted",
    )
    parser.add_argument("--img-server-ip", default="192.168.31.3")
    parser.add_argument("--task-dir", default=str(default_task_dir))
    parser.add_argument("--task-name", default="topstar_h1_mujoco_test")
    parser.add_argument("--task-goal", default="TOPSTAR H1 arm teleoperation")
    parser.add_argument("--task-desc", default="MuJoCo IK Pico teleoperation")
    parser.add_argument(
        "--task-steps",
        default="incremental Pico arm teleoperation",
        help="task steps stored in the legacy episode metadata",
    )
    parser.add_argument(
        "--lerobot-repo-id",
        default=None,
        help="deprecated for live recording; pass the repo id to offline export",
    )
    parser.add_argument("--sync-tolerance-s", type=float, default=1e-4)
    parser.add_argument(
        "--camera-sync-tolerance-s", type=float, default=0.02
    )
    parser.add_argument(
        "--max-camera-age-s",
        type=float,
        default=0.10,
        help="maximum camera age recorded in alignment metadata",
    )
    parser.add_argument(
        "--video-codec",
        default="h264",
        help="deprecated for live recording; pass the codec to offline export",
    )
    parser.add_argument(
        "--dry-run-initial-q",
        type=float,
        nargs=H1_ARM_DOF,
        default=DEFAULT_DRY_RUN_INITIAL_Q,
        metavar="Q",
        help="14-value simulated H1 start pose; ignored with --live",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.live and args.disable_ik_safety_checks:
        raise ValueError("--disable-ik-safety-checks is only allowed in dry-run mode")
    positive = {
        "frequency": args.frequency,
        "arm_scale": args.arm_scale,
        "max_joint_speed": args.max_joint_speed,
        "max_tracking_error": args.max_tracking_error,
        "state_timeout": args.state_timeout,
        "legacy_stale_state_hard_timeout": (
            args.legacy_stale_state_hard_timeout
        ),
        "pico_timeout": args.pico_timeout,
        "max_arm_speed_at_arm": args.max_arm_speed_at_arm,
        "ik_tolerance": args.ik_tolerance,
        "ik_damping": args.ik_damping,
        "max_ik_position_error": args.max_ik_position_error,
        "max_ik_rotation_error_deg": args.max_ik_rotation_error_deg,
    }
    for name, value in positive.items():
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be greater than zero")
    if (
        args.legacy_stale_state_behavior
        and args.legacy_stale_state_hard_timeout <= args.state_timeout
    ):
        raise ValueError(
            "--legacy-stale-state-hard-timeout must be greater than "
            "--state-timeout"
        )
    if args.max_solver_jump is not None and (
        not math.isfinite(args.max_solver_jump) or args.max_solver_jump <= 0.0
    ):
        raise ValueError("--max-solver-jump must be greater than zero")
    if not 0.0 < args.ema_alpha <= 1.0:
        raise ValueError("--ema-alpha must be in (0, 1]")
    if args.position_deadband_m < 0.0 or args.rotation_deadband_deg < 0.0:
        raise ValueError("deadbands must be non-negative")
    if (
        args.ik_max_iters <= 0
        or args.current_q_weight < 0.0
        or args.home_weight < 0.0
    ):
        raise ValueError("IK iteration count and weights are invalid")
    if args.record:
        if not args.task_name.strip():
            raise ValueError("--task-name cannot be empty when recording")
        if not args.raw_record:
            raise ValueError(
                "--no-raw-record is incompatible with raw-only online recording"
            )
        if args.sync_tolerance_s < 0.0 or args.camera_sync_tolerance_s < 0.0:
            raise ValueError("recording tolerances must be non-negative")
        if args.max_camera_age_s < 0.0:
            raise ValueError("--max-camera-age-s must be non-negative")
        if args.raw_record_queue_size <= 0:
            raise ValueError("--raw-record-queue-size must be greater than zero")
    if args.live and args.controller_mapping is None:
        raise ValueError(
            "--live requires an explicit --controller-mapping "
            "{same-side,mirrored}"
        )
    if args.live and args.mujoco_viewer:
        raise ValueError(
            "--mujoco-viewer is disabled with --live; verify mapping in "
            "dry-run, then close the viewer before commanding the robot"
        )


def _state_message(controller: Any) -> Any:
    return controller._ros_node.state_buffer.get()


def _wait_for_robot_state(controller: Any, timeout_s: float = 5.0) -> Any:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        state = _state_message(controller)
        if state is not None and len(state.motor_state) >= 18:
            return state
        time.sleep(0.02)
    raise RuntimeError("no valid /lowstate message received within 5 seconds")


def _seed_robot_command(
    controller: Any, current_q: np.ndarray, state: Any
) -> None:
    """Seed every active LowCmd slot from measurements before arm output."""
    head_q = np.asarray(controller.get_head_q(), dtype=np.float64).reshape(2)
    torso_q = np.array(
        [state.motor_state[0].q, state.motor_state[1].q], dtype=np.float64
    )
    with controller.publish_lock:
        controller.q_target = current_q.copy()
        controller.last_published_q = current_q.copy()
        controller.tauff_target = np.zeros(H1_ARM_DOF, dtype=np.float64)
        controller.head_target = head_q.copy()
        controller.last_published_head_q = head_q.copy()
        controller.torso_target = torso_q.copy()
        controller.last_published_torso = torso_q.copy()


def _start_h1_controller(frequency: float) -> Any:
    """Construct the real H1 controller only after live CLI validation."""
    from teleop.robot_control._base.control_mode import ControlMode
    from teleop.robot_control.topstar_h1.arm_controller import H1ArmController
    from teleop.robot_control.topstar_h1.config import H1RobotConfig

    return H1ArmController(
        H1RobotConfig(),
        ControlMode.ARMS_HEAD,
        frequency=frequency,
        simulation_mode=False,
    )


def _start_televuer() -> Any:
    from televuer import TeleVuerWrapper

    return TeleVuerWrapper(
        use_hand_tracking=False,
        binocular=False,
        img_shape=(480, 640),
        display_mode="pass-through",
        zmq=False,
        webrtc=False,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    _validate_args(args)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    repo_root = Path(__file__).resolve().parents[1]
    arm_limit_mode = (
        "hard" if args.live and args.allow_hard_limit_start else "safe"
    )
    solver = H1MuJoCoLMIK.from_h1_assets(
        repo_root, arm_limit_mode=arm_limit_mode
    )
    LOG.info(
        "H1 model: %s; arm convention: %s",
        H1_MODEL_REVISION,
        H1_ARM_COORDINATE_CONVENTION,
    )
    controller_mapping = args.controller_mapping or "same-side"
    LOG.info("Controller mapping: %s", controller_mapping)
    max_solver_jump = args.max_solver_jump
    if max_solver_jump is None:
        max_solver_jump = 0.35 if args.live else 0.8
    core = H1MuJoCoLiveCore(
        solver,
        position_scale=args.arm_scale,
        ema_alpha=args.ema_alpha,
        position_deadband_m=args.position_deadband_m,
        rotation_deadband_deg=args.rotation_deadband_deg,
        orientation_control=args.orientation_control,
        max_joint_speed_rad_s=args.max_joint_speed,
        max_solver_jump_rad=max_solver_jump,
        max_tracking_error_rad=args.max_tracking_error,
        tracking_error_check_enabled=not args.disable_tracking_error_check,
        max_iters=args.ik_max_iters,
        tolerance=args.ik_tolerance,
        damping=args.ik_damping,
        current_q_weight=args.current_q_weight,
        home_qpos=DEFAULT_DRY_RUN_INITIAL_Q,
        home_weight=args.home_weight,
        max_ik_position_error_m=args.max_ik_position_error,
        max_ik_rotation_error_rad=math.radians(
            args.max_ik_rotation_error_deg
        ),
        hold_on_ik_safety=not args.live,
        ik_safety_checks_enabled=not args.disable_ik_safety_checks,
        mirror_lateral_translation=controller_mapping == "mirrored",
    )
    if args.disable_tracking_error_check:
        LOG.warning(
            "TRACKING ERROR CHECK DISABLED: command/measured arm divergence "
            "will not disarm teleoperation; all other live safeguards remain active"
        )
    if args.legacy_stale_state_behavior:
        LOG.warning(
            "LEGACY STALE-STATE MODE: brief /lowstate gaps reuse the last "
            "valid measured state; hard disarm remains at %.2f seconds",
            args.legacy_stale_state_hard_timeout,
        )
    if args.disable_ik_safety_checks:
        LOG.warning(
            "DRY-RUN IK SAFETY CHECKS DISABLED: non-converged and branch-jump "
            "solutions will be displayed when finite and within mechanical limits"
        )
    LOG.info("End-effector orientation control: %s", args.orientation_control)

    controller = None
    tv_wrapper = None
    image_client = None
    raw_writer = None
    mujoco_viewer: H1MuJoCoViewer | None = None
    recording_session: FormalRecordingSession | None = None
    last_valid_q: np.ndarray | None = None
    last_valid_state_receive_ns = 0
    pico_connected = False
    stop_requested = False

    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    try:
        if args.live:
            LOG.warning("LIVE MODE: commands will be published to TOPSTAR_H1")
            if args.allow_hard_limit_start:
                LOG.warning(
                    "HARD-LIMIT HOLD MODE: the measured pose is captured without "
                    "motion and IK uses mechanical limits instead of the normal "
                    "5-degree margin"
                )
            controller = _start_h1_controller(args.frequency)
            initial_state = _wait_for_robot_state(controller)
            current_q, current_dq, state_receive_ns = _read_robot_arm_state(
                controller
            )
            if not monotonic_timestamp_is_fresh(
                state_receive_ns,
                args.state_timeout,
            ):
                raise LiveSafetyError(
                    "initial /lowstate snapshot is stale; no command was sent"
                )
            current_q, current_dq = _validate_robot_arm_state(
                solver,
                current_q,
                current_dq,
            )
            publishers, subscribers = _validate_ros_command_graph(controller)
            _seed_robot_command(controller, current_q, initial_state)
            last_valid_q = current_q.copy()
            last_valid_state_receive_ns = state_receive_ns
            LOG.info(
                "Robot preflight passed: /lowcmd publishers=%d subscribers=%d, "
                "max |dq|=%.4f rad/s",
                publishers,
                subscribers,
                float(np.max(np.abs(current_dq))),
            )
        else:
            LOG.info("DRY-RUN MODE: no ROS2 controller and no robot output")
            current_q = _as_arm_q(args.dry_run_initial_q, "dry-run initial q")
            current_dq = np.zeros(H1_ARM_DOF, dtype=np.float64)

        if args.mujoco_viewer:
            mujoco_viewer = H1MuJoCoViewer(solver)
            mujoco_viewer.update(current_q)
            LOG.info("MuJoCo viewer opened; close its window to stop visualization")

        if args.record:
            from teleimager.image_client import ImageClient
            from teleop.utils.raw_session_writer import RawSessionWriter

            image_client = ImageClient(
                host=args.img_server_ip, request_bgr=False
            )
            camera_config = image_client.get_cam_config()
            requested_camera_sources = tuple(
                dict.fromkeys(args.record_cameras or CAMERA_SOURCES)
            )
            missing_config = [
                source
                for source in requested_camera_sources
                for config_key in (CAMERA_CONFIG_BY_SOURCE[source],)
                if not (camera_config.get(config_key) or {}).get(
                    "enable_zmq", False
                )
            ]
            if missing_config:
                raise RuntimeError(
                    "requested ZMQ cameras are disabled: "
                    + ", ".join(missing_config)
                )
            camera_config = {
                key: dict(value) if isinstance(value, dict) else value
                for key, value in camera_config.items()
            }
            for source, config_key in CAMERA_CONFIG_BY_SOURCE.items():
                if source not in requested_camera_sources:
                    camera_entry = camera_config.get(config_key)
                    if not isinstance(camera_entry, dict):
                        camera_entry = {}
                        camera_config[config_key] = camera_entry
                    camera_entry["enable_zmq"] = False
            if args.record_cameras is not None:
                LOG.warning(
                    "PARTIAL CAMERA RECORDING: %s",
                    ", ".join(requested_camera_sources),
                )
            camera_key_plan = build_camera_key_plan(camera_config)
            output_root = Path(args.task_dir).expanduser().resolve() / args.task_name
            raw_writer = (
                RawSessionWriter(
                    output_root,
                    queue_size=args.raw_record_queue_size,
                )
                if args.raw_record
                else None
            )
            if raw_writer is not None and controller is not None and hasattr(
                controller, "set_raw_event_sink"
            ):
                controller.set_raw_event_sink(raw_writer.append_event)
            recording_session = FormalRecordingSession(
                image_client=image_client,
                camera_config=camera_config,
                camera_key_plan=camera_key_plan,
                legacy_episode_writer=None,
                output_root=output_root,
                task_name=args.task_name,
                task_description=args.task_desc,
                frequency=args.frequency,
                sync_tolerance_s=args.sync_tolerance_s,
                camera_sync_tolerance_s=args.camera_sync_tolerance_s,
                raw_writer=raw_writer,
                max_camera_age_s=args.max_camera_age_s,
            )
            LOG.info(
                "Recording ready (raw-only online; no ServoJ JSON): "
                "%s; legacy images and LeRobot are generated offline",
                output_root,
            )
            LOG.info(
                "Offline finalize: python -m teleop.utils.finalize_raw_dataset "
                "--task-dir %s",
                output_root,
            )

        tv_wrapper = _start_televuer()
        LOG.info("Open https://<this-computer-ip>:8012 in Pico")
        LOG.info(
            "Left A: arm/disarm both arms; "
            "Right A: lock/unlock rigid box grasp; Ctrl-C: exit"
        )
        if controller_mapping == "mirrored":
            LOG.warning(
                "MIRRORED mapping: Pico left controls robot right; "
                "Pico right controls robot left; lateral translation is reversed"
            )
        else:
            LOG.warning(
                "SAME-SIDE mapping: Pico left controls robot left; "
                "Pico right controls robot right"
            )
        if recording_session is not None:
            LOG.info(
                "Left B/Y: start/stop recording; "
                "Right B/X: discard latest episode"
            )
        LOG.info(
            "Left Grip: return to HOME after releasing it; "
            "HOME is also performed once at startup"
        )

        def _drain_raw_xr_events() -> None:
            if raw_writer is None or tv_wrapper is None:
                return
            drain = getattr(tv_wrapper, "drain_raw_xr_events", None)
            if drain is None:
                return
            for event in drain():
                raw_writer.append_event("pico", event)

        previous_left_a = False
        previous_right_a = False
        previous_left_b = False
        previous_right_b = False
        last_cycle = time.monotonic()
        nominal_dt = 1.0 / args.frequency
        next_cycle = last_cycle
        last_status = 0.0
        reference_settle_deadline = 0.0
        last_invalid_wrist_log = 0.0
        last_invalid_state_log = 0.0
        home_left_grip_armed = False

        def _return_home() -> bool:
            """Run H1ArmController.go_home and rebase the live state."""
            nonlocal current_q, current_dq, state_receive_ns
            nonlocal last_valid_q, last_valid_state_receive_ns
            nonlocal reference_settle_deadline
            if not args.live:
                return True
            if controller is None or not hasattr(controller, "go_home"):
                LOG.error("Cannot start recording: controller has no go_home()")
                return False
            try:
                core.disarm()
                controller.go_home()
                if hasattr(controller, "wait_for_hold_expire"):
                    controller.wait_for_hold_expire(timeout=10.0)
                current_q, current_dq, state_receive_ns = (
                    _wait_for_fresh_robot_arm_state(
                        controller,
                        freshness_timeout_s=args.state_timeout,
                        wait_timeout_s=(
                            args.legacy_stale_state_hard_timeout
                            if args.legacy_stale_state_behavior
                            else max(1.0, args.state_timeout * 2.0)
                        ),
                    )
                )
                current_q, current_dq = _validate_robot_arm_state(
                    solver, current_q, current_dq
                )
                if hasattr(controller, "reset_arm_command_reference"):
                    controller.reset_arm_command_reference(current_q)
                last_valid_q = current_q.copy()
                last_valid_state_receive_ns = state_receive_ns
                reference_settle_deadline = 0.0
                LOG.info("go_home() reached; measured state rebased")
                return True
            except Exception as exc:
                core.disarm()
                LOG.error("go_home() failed: %s", exc)
                return False

        def _go_home_before_recording(
            left_pose: np.ndarray, right_pose: np.ndarray
        ) -> bool:
            """Compatibility helper retained for callers/tests."""
            nonlocal reference_settle_deadline
            if not _return_home():
                return False
            core.arm(current_q, left_pose, right_pose)
            reference_settle_deadline = time.monotonic() + 0.5
            return True

        if args.live:
            if not _return_home():
                raise LiveSafetyError("initial go_home() failed")
            LOG.info("Initial go_home() completed; press Left A to arm teleoperation")

        while not stop_requested:
            now = time.monotonic()
            if now < next_cycle:
                time.sleep(next_cycle - now)
                now = time.monotonic()
            next_cycle = max(next_cycle + nominal_dt, now)
            dt = min(max(now - last_cycle, nominal_dt * 0.25), nominal_dt)
            last_cycle = now

            if args.live:
                try:
                    measured_q, measured_dq, state_receive_ns = (
                        _read_robot_arm_state(controller)
                    )
                    state_is_fresh = monotonic_timestamp_is_fresh(
                        state_receive_ns,
                        args.state_timeout,
                        now_ns=time.monotonic_ns(),
                    )
                    if state_is_fresh:
                        measured_q, measured_dq = _validate_robot_arm_state(
                            solver,
                            measured_q,
                            measured_dq,
                        )
                        last_valid_q = measured_q.copy()
                        last_valid_state_receive_ns = state_receive_ns
                    else:
                        state_age_s = (
                            time.monotonic_ns() - int(state_receive_ns or 0)
                        ) / 1_000_000_000.0
                        can_reuse_legacy_state = (
                            args.legacy_stale_state_behavior
                            and last_valid_q is not None
                            and state_receive_ns > 0
                            and 0.0 <= state_age_s
                            <= args.legacy_stale_state_hard_timeout
                        )
                        if not can_reuse_legacy_state:
                            raise LiveSafetyError(
                                "/lowstate is stale or missing"
                            )
                        measured_q = last_valid_q.copy()
                        measured_dq = np.zeros(H1_ARM_DOF, dtype=np.float64)
                        if now - last_invalid_state_log >= 1.0:
                            last_invalid_state_log = now
                            LOG.warning(
                                "LEGACY stale-state fallback: reusing the last "
                                "valid arm state (age %.3f s)",
                                state_age_s,
                            )
                except Exception as exc:
                    if core.active:
                        core.disarm()
                        if last_valid_q is not None:
                            _send_robot_hold(controller, solver, last_valid_q)
                        LOG.error("DISARMED: invalid /lowstate: %s", exc)
                        if (
                            recording_session is not None
                            and recording_session.active
                        ):
                            try:
                                recording_session.stop()
                                LOG.info("Recording saved after /lowstate timeout")
                            except Exception:
                                LOG.exception(
                                    "failed to save recording after /lowstate timeout"
                                )
                    elif now - last_invalid_state_log >= 1.0:
                        last_invalid_state_log = now
                        LOG.error("Waiting for valid /lowstate: %s", exc)
                    continue
                current_q = measured_q
                current_dq = measured_dq

            if mujoco_viewer is not None:
                if not mujoco_viewer.is_running():
                    LOG.info("MuJoCo viewer was closed")
                    break
                mujoco_viewer.update(current_q)

            process = getattr(getattr(tv_wrapper, "tvuer", None), "process", None)
            if process is not None and not process.is_alive():
                raise RuntimeError("TeleVuer subprocess stopped")
            tele_data = tv_wrapper.get_tele_data()
            _drain_raw_xr_events()
            pico_timestamp_ns = int(
                getattr(tele_data, "controller_pose_timestamp_ns", 0) or 0
            )
            if not monotonic_timestamp_is_fresh(
                pico_timestamp_ns,
                args.pico_timeout,
                now_ns=time.monotonic_ns(),
            ):
                if pico_connected:
                    pico_connected = False
                    LOG.info("Pico controller connection closed")
                if core.active:
                    core.disarm()
                    reference_settle_deadline = 0.0
                    if controller is not None:
                        _send_robot_hold(controller, solver, current_q)
                    LOG.error(
                        "DISARMED: Pico controller data is stale; "
                        "holding measured pose"
                    )
                    if recording_session is not None and recording_session.active:
                        try:
                            recording_session.stop()
                            LOG.info("Recording saved after Pico timeout")
                        except Exception:
                            LOG.exception(
                                "failed to save recording after Pico timeout"
                            )
                continue
            if not pico_connected:
                pico_connected = True
                LOG.info("Pico controller connected")

            wrists_valid = bool(
                getattr(tele_data, "left_wrist_valid", True)
                and getattr(tele_data, "right_wrist_valid", True)
            )
            left_wrist_pose, right_wrist_pose = mapped_wrist_poses(
                tele_data, controller_mapping
            )

            left_a = bool(tele_data.left_ctrl_aButton)
            right_a = bool(tele_data.right_ctrl_aButton)
            left_b = bool(tele_data.left_ctrl_bButton)
            right_b = bool(getattr(tele_data, "right_ctrl_bButton", False))
            left_grip_pressed = bool(
                getattr(tele_data, "left_ctrl_squeeze", False)
            ) or float(
                getattr(tele_data, "left_ctrl_squeezeValue", 0.0) or 0.0
            ) > 0.5
            if not left_grip_pressed:
                home_left_grip_armed = True
            left_grip_rising = left_grip_pressed and home_left_grip_armed
            left_a_rising = left_a and not previous_left_a
            right_a_rising = right_a and not previous_right_a
            left_b_rising = left_b and not previous_left_b
            right_b_rising = right_b and not previous_right_b
            previous_left_a = left_a
            previous_right_a = right_a
            previous_left_b = left_b
            previous_right_b = right_b

            if left_grip_rising:
                home_left_grip_armed = False
                if recording_session is not None and recording_session.active:
                    LOG.warning(
                        "Left Grip HOME ignored: stop the active recording first"
                    )
                elif core.active:
                    LOG.warning("Left Grip HOME ignored: stop teleoperation first")
                elif _return_home():
                    LOG.info("Left Grip HOME completed")
                continue

            if right_a_rising:
                if not core.active:
                    LOG.warning("Right A box lock ignored: arm teleoperation is not active")
                    continue
                if now < reference_settle_deadline:
                    LOG.warning(
                        "Right A box lock ignored: wrist references are still stabilizing"
                    )
                    continue
                max_toggle_speed = float(np.max(np.abs(current_dq)))
                if max_toggle_speed > 0.75:
                    LOG.warning(
                        "Right A box lock ignored: max measured |dq| %.3f rad/s "
                        "exceeds 0.750 rad/s",
                        max_toggle_speed,
                    )
                    continue
                try:
                    locked, lock_width = core.toggle_dual_object_lock(
                        current_q,
                        left_wrist_pose,
                        right_wrist_pose,
                        now_s=now,
                    )
                    if controller is not None and hasattr(
                        controller, "reset_arm_command_reference"
                    ):
                        controller.reset_arm_command_reference(current_q)
                    if locked:
                        LOG.warning(
                            "BOX GRASP LOCKED by Right A: measured width %.3f m",
                            lock_width,
                        )
                    else:
                        LOG.warning(
                            "BOX GRASP UNLOCKED by Right A: independent arm references recaptured"
                        )
                except (LiveSafetyError, ValueError) as exc:
                    LOG.error("Right A box lock rejected: %s", exc)
                continue

            if left_a_rising:
                if core.active:
                    core.disarm()
                    reference_settle_deadline = 0.0
                    if controller is not None:
                        _send_robot_hold(controller, solver, current_q)
                    LOG.warning("DISARMED by Left A; holding measured pose")
                    if recording_session is not None and recording_session.active:
                        try:
                            recording_session.stop()
                            LOG.info("Recording saved after disarm")
                        except Exception:
                            LOG.exception("failed to save recording after disarm")
                else:
                    if not wrists_valid:
                        LOG.warning(
                            "Arm request ignored: Pico controller pose is not valid yet"
                        )
                        continue
                    if args.live and float(np.max(np.abs(current_dq))) > (
                        args.max_arm_speed_at_arm
                    ):
                        LOG.warning(
                            "Arm request ignored: max measured |dq| %.4f rad/s "
                            "exceeds %.4f rad/s",
                            float(np.max(np.abs(current_dq))),
                            args.max_arm_speed_at_arm,
                        )
                        continue
                    with (
                        mujoco_viewer.lock()
                        if mujoco_viewer is not None
                        else nullcontext()
                    ):
                        core.arm(
                            current_q,
                            left_wrist_pose,
                            right_wrist_pose,
                        )
                    reference_settle_deadline = now + 0.5
                    LOG.warning(
                        "ARMED: stabilizing Pico wrist references for 0.5 seconds"
                    )
                continue

            if left_b_rising and recording_session is not None:
                if recording_session.active:
                    recording_session.stop()
                    LOG.info("Recording episode saved")
                elif not core.active or now < reference_settle_deadline:
                    LOG.error("Recording ignored: arm teleoperation is not active")
                else:
                    try:
                        recording_session.start()
                        LOG.warning(
                            "RECORDING STARTED: three camera frames verified"
                        )
                    except Exception as exc:
                        LOG.error("Recording did not start: %s", exc)
                continue

            if right_b_rising and recording_session is not None:
                try:
                    if recording_session.active:
                        recording_session.discard()
                        LOG.warning("Recording episode discarded")
                    else:
                        recording_session.discard_latest()
                        LOG.warning("Latest recording episode discarded")
                except Exception as exc:
                    LOG.error("Recording discard failed: %s", exc)
                continue

            if not core.active:
                continue

            if not wrists_valid:
                if args.live:
                    core.disarm()
                    if controller is not None:
                        _send_robot_hold(controller, solver, current_q)
                    LOG.error(
                        "DISARMED: Pico controller pose became invalid; "
                        "holding measured pose"
                    )
                elif now - last_invalid_wrist_log >= 1.0:
                    last_invalid_wrist_log = now
                    LOG.warning(
                        "Pico controller pose is invalid; dry-run is holding "
                        "the last MuJoCo pose"
                    )
                continue

            if now < reference_settle_deadline:
                # Pico can report a stale pose for a few frames around a button
                # press. Re-capture until it settles so that sample cannot become
                # the fixed teleoperation reference.
                with (
                    mujoco_viewer.lock()
                    if mujoco_viewer is not None
                    else nullcontext()
                ):
                    core.arm(
                        current_q,
                        left_wrist_pose,
                        right_wrist_pose,
                    )
                continue

            camera_batch = None
            if recording_session is not None and recording_session.active:
                camera_batch = recording_session.capture_camera_batch()
            control_cycle_timestamp_ns = time.time_ns()
            measured_q_for_frame = current_q.copy()
            try:
                with (
                    mujoco_viewer.lock()
                    if mujoco_viewer is not None
                    else nullcontext()
                ):
                    step_result = core.step(
                        current_q,
                        left_wrist_pose,
                        right_wrist_pose,
                        dt,
                    )
            except (LiveSafetyError, ValueError, np.linalg.LinAlgError) as exc:
                core.disarm()
                if controller is not None:
                    _send_robot_hold(controller, solver, current_q)
                LOG.error("DISARMED by safety check: %s", exc)
                if recording_session is not None and recording_session.active:
                    try:
                        recording_session.stop()
                        LOG.info("Recording saved after safety disarm")
                    except Exception:
                        LOG.exception("failed to save recording after safety disarm")
                continue

            current_q = step_result.command_q.copy() if not args.live else current_q
            if controller is not None:
                controller.servo_dual_arm(
                    step_result.command_q,
                    solver.gravity_compensation(step_result.command_q),
                )
            if mujoco_viewer is not None:
                mujoco_viewer.update(step_result.command_q)
            robot_step_done_timestamp_ns = time.time_ns()

            if recording_session is not None and recording_session.active:
                recording_session.add_frame(
                    camera_batch=camera_batch,
                    measured_q=measured_q_for_frame,
                    command_q=step_result.command_q,
                    raw_ik_q=step_result.raw_ik_q,
                    ik_result=step_result.ik,
                    accepted_approximately=step_result.accepted_approximately,
                    control_cycle_timestamp_ns=control_cycle_timestamp_ns,
                    robot_step_done_timestamp_ns=robot_step_done_timestamp_ns,
                    pico_timestamp_ns=pico_timestamp_ns,
                    state_receive_timestamp_ns=(
                        state_receive_ns if args.live else None
                    ),
                )

            if now - last_status >= 1.0:
                last_status = now
                LOG.info(
                    "active%s%s | IK iters=%d pos_err=%.4f m rot_err=%.4f rad",
                    " + recording"
                    if recording_session is not None
                    and recording_session.active
                    else "",
                    " + holding last safe target"
                    if step_result.held_for_ik_safety
                    else "",
                    step_result.ik.iterations,
                    step_result.ik.translation_error_norm,
                    step_result.ik.rotation_error_norm,
                )
    finally:
        if core.active:
            core.disarm()
        if (
            controller is not None
            and last_valid_q is not None
            and bool(getattr(controller, "_arm_active", False))
        ):
            if monotonic_timestamp_is_fresh(
                last_valid_state_receive_ns, args.state_timeout
            ):
                try:
                    _send_robot_hold(controller, solver, last_valid_q)
                    time.sleep(0.1)
                except Exception:
                    LOG.exception("failed to send final hold command")
            else:
                LOG.warning(
                    "Skipped final hold because the last /lowstate is stale"
                )
        if recording_session is not None:
            try:
                recording_session.close()
            except Exception:
                LOG.exception("failed to finalize recording")
        if controller is not None:
            controller.stop()
        if pico_connected:
            pico_connected = False
            LOG.info("Pico controller connection closed")
        if tv_wrapper is not None:
            tv_wrapper.close()
        if image_client is not None:
            image_client.close()
        if mujoco_viewer is not None:
            mujoco_viewer.close()
        LOG.info("Stopped without automatic go-home")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
