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
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from teleop.robot_control.vr_mujoco_relative_teleop import (
    H1_XR_TO_ROBOT_ROTATION,
    H1MuJoCoLMIK,
    LMIKResult,
    VRRelativePoseTracker,
)


LOG = logging.getLogger("h1_vr_mujoco_live")
H1_ARM_DOF = 14
CAMERA_SOURCES = ("head", "left_wrist", "right_wrist")
# A verified non-singular H1 arms-only pose from topstar_h1_test_004.  Dry-run
# uses this instead of all zeros so the viewer starts from a useful IK posture.
DEFAULT_DRY_RUN_INITIAL_Q = [
    -1.2305397,
    -1.1173229,
    -0.76580536,
    0.7694363,
    0.54479694,
    -0.348951,
    0.04157807,
    1.5395422,
    -1.3764126,
    0.25574026,
    -0.34151757,
    -0.25700444,
    0.34901664,
    -0.05125465,
]


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
        max_joint_speed_rad_s: float = 0.5,
        max_solver_jump_rad: float = 0.35,
        max_tracking_error_rad: float = 0.35,
        max_iters: int = 60,
        tolerance: float = 1e-4,
        damping: float = 1e-3,
        current_q_weight: float = 0.01,
        home_qpos: Sequence[float] | None = None,
        home_weight: float = 0.0,
        max_ik_position_error_m: float = 0.06,
        max_ik_rotation_error_rad: float = math.radians(5.0),
        hold_on_ik_safety: bool = False,
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
        self.max_joint_speed_rad_s = max_joint_speed_rad_s
        self.max_solver_jump_rad = max_solver_jump_rad
        self.max_tracking_error_rad = max_tracking_error_rad
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
        self.left_tracker: VRRelativePoseTracker | None = None
        self.right_tracker: VRRelativePoseTracker | None = None
        self.previous_command_q: np.ndarray | None = None
        self.previous_raw_ik_q: np.ndarray | None = None
        self.active = False

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
        self.active = True

    def disarm(self) -> None:
        self.active = False
        self.left_tracker = None
        self.right_tracker = None
        self.previous_command_q = None
        self.previous_raw_ik_q = None
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
        if tracking_error > self.max_tracking_error_rad:
            raise LiveSafetyError(
                f"robot tracking error {tracking_error:.4f} rad exceeds "
                f"{self.max_tracking_error_rad:.4f} rad"
            )
        left_target = self.left_tracker.update(left_wrist_pose)
        right_target = self.right_tracker.update(right_wrist_pose)
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
            if (
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
        # Compare consecutive IK solutions, not the speed-limited command. A
        # user can legitimately move faster than the configured robot speed,
        # in which case the command lags the target by design.
        solver_jump = float(np.max(np.abs(raw_ik_q - self.previous_raw_ik_q)))
        if solver_jump > self.max_solver_jump_rad:
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
        self.previous_command_q = command_q.copy()
        self.previous_raw_ik_q = raw_ik_q.copy()
        return LiveStepResult(
            command_q, raw_ik_q, result, accepted_approximately
        )


class FormalRecordingSession:
    """Keep LeRobot video data and legacy JSON on the same control frames."""

    def __init__(
        self,
        *,
        image_client: Any,
        camera_config: dict[str, Any],
        camera_key_plan: list[dict[str, Any]],
        episode_writer: Any,
        recorder_cls: Any,
        output_root: Path,
        task_name: str,
        task_description: str,
        frequency: float,
        sync_tolerance_s: float,
        camera_sync_tolerance_s: float,
    ) -> None:
        self.image_client = image_client
        self.camera_config = camera_config
        self.camera_key_plan = camera_key_plan
        self.episode_writer = episode_writer
        self.recorder_cls = recorder_cls
        self.output_root = output_root
        self.task_name = task_name
        self.task_description = task_description
        self.frequency = frequency
        self.sync_tolerance_s = sync_tolerance_s
        self.camera_sync_tolerance_s = camera_sync_tolerance_s
        self.active = False
        self.episode_index = 0
        self.frame_index = 0
        self.legacy_recorder: Any = None
        self._prefetched_images: dict[str, Any] | None = None

    def _read_images(self) -> dict[str, Any]:
        return {
            "head": self.image_client.get_head_frame(),
            "left_wrist": self.image_client.get_left_wrist_frame(),
            "right_wrist": self.image_client.get_right_wrist_frame(),
        }

    def _missing_physical_cameras(self, images: dict[str, Any]) -> list[str]:
        missing = []
        for source in CAMERA_SOURCES:
            image = images.get(source)
            if image is None or getattr(image, "bgr", None) is None:
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

        legacy_name = (
            self.task_name
            if self.episode_index == 0
            else f"{self.task_name}_episode_{self.episode_index:03d}"
        )
        recorder = self.recorder_cls(buffer_size=1000)
        recorder.start_program(
            legacy_name,
            self.task_description,
            filepath=str(self.output_root),
            frequency=int(round(self.frequency)),
            control_mode="arms_only",
            robot_model="TOPSTAR_H1_MUJOCO_IK",
        )
        recorder.start_servo_recording()
        try:
            if not self.episode_writer.create_episode():
                raise RuntimeError("LeRobot episode is already active")
        except Exception:
            recorder.stop_servo_recording(smooth_window_size=1)
            recorder.finish()
            raise

        self.legacy_recorder = recorder
        self._prefetched_images = images
        self.frame_index = 0
        self.active = True

    def capture_camera_batch(self) -> CameraBatch:
        if not self.active:
            raise RuntimeError("recording is not active")
        images = self._prefetched_images or self._read_images()
        self._prefetched_images = None
        return extract_camera_batch(
            self.camera_config, self.camera_key_plan, images
        )

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
    ) -> None:
        if not self.active or self.legacy_recorder is None:
            raise RuntimeError("recording is not active")
        states, actions = build_lerobot_state_action(measured_q, command_q)
        camera_timestamps = list(camera_batch.timestamps_ns.values())
        camera_sync_skew_s = None
        if camera_timestamps:
            camera_sync_skew_s = (
                max(camera_timestamps) - min(camera_timestamps)
            ) / 1_000_000_000.0
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
        self.episode_writer.add_item(
            colors=camera_batch.colors,
            depths=camera_batch.depths,
            states=states,
            actions=actions,
            alignment=alignment,
        )

        command_deg = np.rad2deg(
            _as_arm_q(command_q, "command_q")
        ).tolist()
        measured_deg = np.rad2deg(
            _as_arm_q(measured_q, "measured_q")
        ).tolist()
        raw_ik_deg = np.rad2deg(_as_arm_q(raw_ik_q, "raw_ik_q")).tolist()
        self.legacy_recorder.add_ServoJ(
            command_deg,
            state_joint_pose=measured_deg,
            raw_ik_joint_pose=raw_ik_deg,
            frame_index=self.frame_index,
        )
        self.frame_index += 1

    def stop(self) -> None:
        if not self.active:
            return
        errors = []
        try:
            self.episode_writer.save_episode()
        except Exception as exc:
            errors.append(f"LeRobot save failed: {exc}")
        try:
            if self.legacy_recorder is not None:
                if self.legacy_recorder._servo_recording:
                    self.legacy_recorder.stop_servo_recording(
                        smooth_window_size=1
                    )
                self.legacy_recorder.finish()
        except Exception as exc:
            errors.append(f"legacy JSON save failed: {exc}")
        self.active = False
        self.legacy_recorder = None
        self._prefetched_images = None
        self.episode_index += 1
        if errors:
            raise RuntimeError("; ".join(errors))

    def close(self) -> None:
        stop_error = None
        try:
            self.stop()
        except Exception as exc:
            stop_error = exc
        self.episode_writer.close()
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
        self._viewer = mujoco.viewer.launch_passive(
            solver._model,
            solver._data,
            show_left_ui=True,
            show_right_ui=True,
        )

    def is_running(self) -> bool:
        return self._viewer.is_running()

    def update(self, arm_q: Sequence[float]) -> None:
        self._solver.forward_kinematics(_as_arm_q(arm_q, "viewer arm_q"))
        self._viewer.sync()

    def close(self) -> None:
        self._viewer.close()


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
    parser.add_argument("--frequency", type=float, default=20.0)
    parser.add_argument("--arm-scale", type=float, default=1.0)
    parser.add_argument("--ema-alpha", type=float, default=0.8)
    parser.add_argument("--position-deadband-m", type=float, default=0.01)
    parser.add_argument("--rotation-deadband-deg", type=float, default=3.0)
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
    parser.add_argument("--state-timeout", type=float, default=0.5)
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
            "held/rejected. Raised from 0.04 to give --current-q-weight's "
            "steady-state residual headroom before it trips the safety hold"
        ),
    )
    parser.add_argument("--max-ik-rotation-error-deg", type=float, default=5.0)
    parser.add_argument("--ik-damping", type=float, default=1e-3)
    parser.add_argument(
        "--current-q-weight",
        type=float,
        default=0.01,
        help=(
            "joint-space pullback toward the solve's starting arm state; "
            "constrains the redundant 7-DOF null space so the elbow cannot flip "
            "between IK branches on nearly identical targets (the main cause of "
            "large IK joint jumps). Any value above ~0 raises the IK's steady-"
            "state residual (it trades exact task convergence for null-space "
            "stability), so keep this small — larger values push the residual "
            "toward --max-ik-position-error and trigger holds sooner. 0 "
            "disables it"
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
        help="record synchronized LeRobot videos/state/action and legacy JSON",
    )
    parser.add_argument("--img-server-ip", default="192.168.31.3")
    parser.add_argument("--task-dir", default=str(default_task_dir))
    parser.add_argument("--task-name", default="topstar_h1_mujoco_test")
    parser.add_argument("--task-goal", default="TOPSTAR H1 arm teleoperation")
    parser.add_argument("--task-desc", default="MuJoCo IK Pico teleoperation")
    parser.add_argument("--lerobot-repo-id", default=None)
    parser.add_argument("--sync-tolerance-s", type=float, default=1e-4)
    parser.add_argument(
        "--camera-sync-tolerance-s", type=float, default=0.02
    )
    parser.add_argument("--video-codec", default="h264")
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
    positive = {
        "frequency": args.frequency,
        "arm_scale": args.arm_scale,
        "max_joint_speed": args.max_joint_speed,
        "max_tracking_error": args.max_tracking_error,
        "state_timeout": args.state_timeout,
        "ik_tolerance": args.ik_tolerance,
        "ik_damping": args.ik_damping,
        "max_ik_position_error": args.max_ik_position_error,
        "max_ik_rotation_error_deg": args.max_ik_rotation_error_deg,
    }
    for name, value in positive.items():
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be greater than zero")
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
        if args.sync_tolerance_s < 0.0 or args.camera_sync_tolerance_s < 0.0:
            raise ValueError("recording tolerances must be non-negative")
        if not args.video_codec.strip():
            raise ValueError("--video-codec cannot be empty")


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
    solver = H1MuJoCoLMIK.from_h1_assets(repo_root)
    max_solver_jump = args.max_solver_jump
    if max_solver_jump is None:
        max_solver_jump = 0.35 if args.live else 0.8
    core = H1MuJoCoLiveCore(
        solver,
        position_scale=args.arm_scale,
        ema_alpha=args.ema_alpha,
        position_deadband_m=args.position_deadband_m,
        rotation_deadband_deg=args.rotation_deadband_deg,
        max_joint_speed_rad_s=args.max_joint_speed,
        max_solver_jump_rad=max_solver_jump,
        max_tracking_error_rad=args.max_tracking_error,
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
    )

    controller = None
    tv_wrapper = None
    image_client = None
    mujoco_viewer: H1MuJoCoViewer | None = None
    recording_session: FormalRecordingSession | None = None
    stop_requested = False

    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    try:
        if args.live:
            from teleop.robot_control._base.control_mode import ControlMode
            from teleop.robot_control.topstar_h1.arm_controller import H1ArmController
            from teleop.robot_control.topstar_h1.config import H1RobotConfig

            LOG.warning("LIVE MODE: commands will be published to TOPSTAR_H1")
            controller = H1ArmController(
                H1RobotConfig(),
                ControlMode.ARMS_HEAD,
                frequency=args.frequency,
                simulation_mode=False,
            )
            initial_state = _wait_for_robot_state(controller)
            current_q = controller.get_current_dual_arm_q()
            _seed_robot_command(controller, current_q, initial_state)
        else:
            LOG.info("DRY-RUN MODE: no ROS2 controller and no robot output")
            current_q = _as_arm_q(args.dry_run_initial_q, "dry-run initial q")

        if args.mujoco_viewer:
            mujoco_viewer = H1MuJoCoViewer(solver)
            mujoco_viewer.update(current_q)
            LOG.info("MuJoCo viewer opened; close its window to stop visualization")

        if args.record:
            from teleimager.image_client import ImageClient
            from teleop.utils.lerobot_episode_writer import (
                LeRobotEpisodeWriter,
                build_camera_key_plan,
                default_repo_id,
            )
            from teleop.utils.record import Recorder

            image_client = ImageClient(
                host=args.img_server_ip, request_bgr=True
            )
            camera_config = image_client.get_cam_config()
            missing_config = [
                source
                for config_key, source in (
                    ("head_camera", "head"),
                    ("left_wrist_camera", "left_wrist"),
                    ("right_wrist_camera", "right_wrist"),
                )
                if not (camera_config.get(config_key) or {}).get(
                    "enable_zmq", False
                )
            ]
            if missing_config:
                raise RuntimeError(
                    "formal recording requires three ZMQ cameras; disabled: "
                    + ", ".join(missing_config)
                )
            camera_key_plan = build_camera_key_plan(camera_config)
            expected_color_keys = [
                entry["color_key"] for entry in camera_key_plan
            ]
            expected_depth_keys = [
                entry["depth_key"]
                for entry in camera_key_plan
                if entry["depth_key"]
            ]
            output_root = Path(args.task_dir).expanduser().resolve() / args.task_name
            episode_writer = LeRobotEpisodeWriter(
                root=output_root / "lerobot",
                repo_id=args.lerobot_repo_id
                or default_repo_id(args.task_name),
                fps=args.frequency,
                task=args.task_goal or args.task_name,
                robot_type="TOPSTAR_H1_MUJOCO_IK",
                tolerance_s=args.sync_tolerance_s,
                use_videos=True,
                expected_color_keys=expected_color_keys,
                expected_depth_keys=expected_depth_keys,
                vcodec=args.video_codec,
            )
            recording_session = FormalRecordingSession(
                image_client=image_client,
                camera_config=camera_config,
                camera_key_plan=camera_key_plan,
                episode_writer=episode_writer,
                recorder_cls=Recorder,
                output_root=output_root,
                task_name=args.task_name,
                task_description=args.task_desc,
                frequency=args.frequency,
                sync_tolerance_s=args.sync_tolerance_s,
                camera_sync_tolerance_s=args.camera_sync_tolerance_s,
            )
            LOG.info("Recording ready: %s", output_root)

        tv_wrapper = _start_televuer()
        LOG.info("Open https://<this-computer-ip>:8012 in Pico")
        LOG.info("Left A: arm/disarm both arms; Right A: disarm; Ctrl-C: exit")
        if recording_session is not None:
            LOG.info("Left B/Y: start/stop and save one recording episode")

        freshness = StateFreshnessMonitor(args.state_timeout)
        previous_left_a = False
        previous_right_a = False
        previous_left_b = False
        last_cycle = time.monotonic()
        nominal_dt = 1.0 / args.frequency
        next_cycle = last_cycle
        last_status = 0.0
        reference_settle_deadline = 0.0
        last_invalid_wrist_log = 0.0

        while not stop_requested:
            now = time.monotonic()
            if now < next_cycle:
                time.sleep(next_cycle - now)
                now = time.monotonic()
            next_cycle = max(next_cycle + nominal_dt, now)
            dt = min(max(now - last_cycle, nominal_dt * 0.25), nominal_dt)
            last_cycle = now

            if args.live:
                state = _state_message(controller)
                if not freshness.observe(state, now):
                    if core.active:
                        core.disarm()
                        LOG.error("DISARMED: /lowstate is stale or missing")
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
                    continue
                current_q = controller.get_current_dual_arm_q()

            if mujoco_viewer is not None:
                if not mujoco_viewer.is_running():
                    LOG.info("MuJoCo viewer was closed")
                    break
                mujoco_viewer.update(current_q)

            process = getattr(getattr(tv_wrapper, "tvuer", None), "process", None)
            if process is not None and not process.is_alive():
                raise RuntimeError("TeleVuer subprocess stopped")
            tele_data = tv_wrapper.get_tele_data()
            wrists_valid = bool(
                getattr(tele_data, "left_wrist_valid", True)
                and getattr(tele_data, "right_wrist_valid", True)
            )

            left_a = bool(tele_data.left_ctrl_aButton)
            right_a = bool(tele_data.right_ctrl_aButton)
            left_b = bool(tele_data.left_ctrl_bButton)
            left_a_rising = left_a and not previous_left_a
            right_a_rising = right_a and not previous_right_a
            left_b_rising = left_b and not previous_left_b
            previous_left_a = left_a
            previous_right_a = right_a
            previous_left_b = left_b

            if right_a_rising and core.active:
                core.disarm()
                reference_settle_deadline = 0.0
                if controller is not None:
                    controller.servo_dual_arm(current_q, np.zeros(H1_ARM_DOF))
                LOG.warning("DISARMED by Right A; holding measured pose")
                if recording_session is not None and recording_session.active:
                    try:
                        recording_session.stop()
                        LOG.info("Recording saved after disarm")
                    except Exception:
                        LOG.exception("failed to save recording after disarm")
                continue

            if left_a_rising:
                if core.active:
                    core.disarm()
                    reference_settle_deadline = 0.0
                    if controller is not None:
                        controller.servo_dual_arm(current_q, np.zeros(H1_ARM_DOF))
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
                    core.arm(
                        current_q,
                        tele_data.left_wrist_pose,
                        tele_data.right_wrist_pose,
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
                        LOG.warning("RECORDING STARTED: three camera frames verified")
                    except Exception as exc:
                        LOG.error("Recording did not start: %s", exc)
                continue

            if not core.active:
                continue

            if not wrists_valid:
                if args.live:
                    core.disarm()
                    if controller is not None:
                        controller.servo_dual_arm(
                            current_q, np.zeros(H1_ARM_DOF)
                        )
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
                core.arm(
                    current_q,
                    tele_data.left_wrist_pose,
                    tele_data.right_wrist_pose,
                )
                continue

            camera_batch = None
            if recording_session is not None and recording_session.active:
                camera_batch = recording_session.capture_camera_batch()
            control_cycle_timestamp_ns = time.time_ns()
            measured_q_for_frame = current_q.copy()
            try:
                step_result = core.step(
                    current_q,
                    tele_data.left_wrist_pose,
                    tele_data.right_wrist_pose,
                    dt,
                )
            except (LiveSafetyError, ValueError, np.linalg.LinAlgError) as exc:
                core.disarm()
                if controller is not None:
                    controller.servo_dual_arm(current_q, np.zeros(H1_ARM_DOF))
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
                    step_result.command_q, np.zeros(H1_ARM_DOF)
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
        if controller is not None:
            try:
                measured = controller.get_current_dual_arm_q()
                controller.servo_dual_arm(measured, np.zeros(H1_ARM_DOF))
                time.sleep(0.1)
            except Exception:
                LOG.exception("failed to send final hold command")
        if recording_session is not None:
            try:
                recording_session.close()
            except Exception:
                LOG.exception("failed to finalize recording")
        if controller is not None:
            controller.stop()
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
