from __future__ import annotations

from collections.abc import Callable
from typing import Optional

import numpy as np


Pose = np.ndarray
TargetValidator = Callable[[Pose, Pose], bool]


def _validated_pose(pose: Pose, name: str) -> Pose:
    value = np.asarray(pose, dtype=float)
    if value.shape != (4, 4):
        raise ValueError(f"{name} must have shape (4, 4), got {value.shape}")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} contains non-finite values")
    if not np.allclose(value[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
        raise ValueError(f"{name} is not a homogeneous transform")
    result = value.copy()
    result[:3, :3] = _project_rotation(result[:3, :3])
    return result


def _project_rotation(rotation: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(np.asarray(rotation, dtype=float))
    projected = u @ vt
    if np.linalg.det(projected) < 0.0:
        u[:, -1] *= -1.0
        projected = u @ vt
    return projected


def _rotation_vector(rotation: np.ndarray) -> np.ndarray:
    rotation = _project_rotation(rotation)
    cosine = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    angle = float(np.arccos(cosine))
    if angle < 1e-8:
        return 0.5 * np.array([
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ])
    if np.pi - angle < 1e-5:
        eigenvalues, eigenvectors = np.linalg.eig(rotation)
        axis_index = int(np.argmin(np.abs(eigenvalues - 1.0)))
        axis = np.real(eigenvectors[:, axis_index])
        norm = float(np.linalg.norm(axis))
        if norm < 1e-8:
            return np.zeros(3)
        return axis / norm * angle
    axis = np.array([
        rotation[2, 1] - rotation[1, 2],
        rotation[0, 2] - rotation[2, 0],
        rotation[1, 0] - rotation[0, 1],
    ]) / (2.0 * np.sin(angle))
    return axis * angle


def _rotation_from_vector(rotation_vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(rotation_vector, dtype=float).reshape(3)
    angle = float(np.linalg.norm(vector))
    if angle < 1e-10:
        skew = np.array([
            [0.0, -vector[2], vector[1]],
            [vector[2], 0.0, -vector[0]],
            [-vector[1], vector[0], 0.0],
        ])
        return np.eye(3) + skew
    axis = vector / angle
    skew = np.array([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0],
    ])
    return np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)


def _limit_vector(
    vector: np.ndarray,
    maximum_norm: Optional[float],
) -> np.ndarray:
    if maximum_norm is None:
        return vector
    norm = float(np.linalg.norm(vector))
    if norm <= maximum_norm or norm < 1e-12:
        return vector
    return vector * (maximum_norm / norm)


class DualObjectController:
    """Generate a rigid pair of end-effector targets from two XR poses.

    Lock and unlock always use measured FK poses as robot-side references.
    While locked, differential hand translation and rotation are rejected;
    only common motion is applied to the captured rigid grasp geometry.
    """

    def __init__(
        self,
        max_translation_speed_m_s: Optional[float] = 0.20,
        max_rotation_speed_rad_s: Optional[float] = np.deg2rad(45.0),
        max_rotation_disagreement_rad: float = np.deg2rad(25.0),
    ):
        self.max_translation_speed_m_s = (
            None
            if max_translation_speed_m_s is None
            else float(max_translation_speed_m_s)
        )
        self.max_rotation_speed_rad_s = (
            None
            if max_rotation_speed_rad_s is None
            else float(max_rotation_speed_rad_s)
        )
        self.max_rotation_disagreement_rad = float(max_rotation_disagreement_rad)
        self.reset()

    @property
    def locked(self) -> bool:
        return self._locked

    @property
    def lock_width_m(self) -> Optional[float]:
        return self._lock_width_m

    def reset(self) -> None:
        self._locked = False
        self._input_left_lock: Optional[Pose] = None
        self._input_right_lock: Optional[Pose] = None
        self._robot_left_lock: Optional[Pose] = None
        self._robot_right_lock: Optional[Pose] = None
        self._robot_center_lock: Optional[np.ndarray] = None
        self._robot_left_offset: Optional[np.ndarray] = None
        self._robot_right_offset: Optional[np.ndarray] = None
        self._command_center: Optional[np.ndarray] = None
        self._command_rotation = np.eye(3)
        self._rotation_input_left_ref: Optional[np.ndarray] = None
        self._rotation_input_right_ref: Optional[np.ndarray] = None
        self._rotation_command_ref = np.eye(3)
        self._rotation_suspended = False
        self._last_update_s: Optional[float] = None
        self._lock_width_m: Optional[float] = None
        self._free_input_left_ref: Optional[Pose] = None
        self._free_input_right_ref: Optional[Pose] = None
        self._free_robot_left_ref: Optional[Pose] = None
        self._free_robot_right_ref: Optional[Pose] = None
        self._free_last_left: Optional[Pose] = None
        self._free_last_right: Optional[Pose] = None
        self.last_rotation_disagreement_rad = 0.0
        self.last_workspace_limited = False

    def lock(
        self,
        input_left: Pose,
        input_right: Pose,
        robot_left_actual: Pose,
        robot_right_actual: Pose,
        now_s: float,
    ) -> float:
        self._input_left_lock = _validated_pose(input_left, "input_left")
        self._input_right_lock = _validated_pose(input_right, "input_right")
        self._robot_left_lock = _validated_pose(robot_left_actual, "robot_left_actual")
        self._robot_right_lock = _validated_pose(robot_right_actual, "robot_right_actual")
        self._robot_center_lock = 0.5 * (
            self._robot_left_lock[:3, 3] + self._robot_right_lock[:3, 3]
        )
        self._robot_left_offset = self._robot_left_lock[:3, 3] - self._robot_center_lock
        self._robot_right_offset = self._robot_right_lock[:3, 3] - self._robot_center_lock
        self._command_center = self._robot_center_lock.copy()
        self._command_rotation = np.eye(3)
        self._rotation_input_left_ref = self._input_left_lock[:3, :3].copy()
        self._rotation_input_right_ref = self._input_right_lock[:3, :3].copy()
        self._rotation_command_ref = np.eye(3)
        self._rotation_suspended = False
        self._last_update_s = float(now_s)
        self._lock_width_m = float(np.linalg.norm(
            self._robot_right_lock[:3, 3] - self._robot_left_lock[:3, 3]
        ))
        self._free_input_left_ref = None
        self._free_input_right_ref = None
        self._free_robot_left_ref = None
        self._free_robot_right_ref = None
        self._free_last_left = None
        self._free_last_right = None
        self.last_rotation_disagreement_rad = 0.0
        self.last_workspace_limited = False
        self._locked = True
        return self._lock_width_m

    def unlock(
        self,
        input_left: Pose,
        input_right: Pose,
        robot_left_actual: Pose,
        robot_right_actual: Pose,
    ) -> None:
        self._free_input_left_ref = _validated_pose(input_left, "input_left")
        self._free_input_right_ref = _validated_pose(input_right, "input_right")
        self._free_robot_left_ref = _validated_pose(robot_left_actual, "robot_left_actual")
        self._free_robot_right_ref = _validated_pose(robot_right_actual, "robot_right_actual")
        self._free_last_left = self._free_robot_left_ref.copy()
        self._free_last_right = self._free_robot_right_ref.copy()
        self._locked = False
        self._last_update_s = None
        self._lock_width_m = None
        self._rotation_suspended = False
        self.last_rotation_disagreement_rad = 0.0
        self.last_workspace_limited = False

    def targets(
        self,
        input_left: Pose,
        input_right: Pose,
        now_s: float,
        validator: Optional[TargetValidator] = None,
    ) -> tuple[Pose, Pose]:
        input_left = _validated_pose(input_left, "input_left")
        input_right = _validated_pose(input_right, "input_right")
        if self._locked:
            return self._locked_targets(input_left, input_right, float(now_s), validator)
        return self._free_targets(input_left, input_right, validator)

    def status(self) -> dict[str, object]:
        return {
            "locked": self._locked,
            "lock_width_m": self._lock_width_m,
            "rotation_disagreement_rad": self.last_rotation_disagreement_rad,
            "rotation_suspended": self._rotation_suspended,
            "workspace_limited": self.last_workspace_limited,
        }

    def _locked_targets(
        self,
        input_left: Pose,
        input_right: Pose,
        now_s: float,
        validator: Optional[TargetValidator],
    ) -> tuple[Pose, Pose]:
        assert self._input_left_lock is not None
        assert self._input_right_lock is not None
        assert self._robot_left_lock is not None
        assert self._robot_right_lock is not None
        assert self._robot_center_lock is not None
        assert self._robot_left_offset is not None
        assert self._robot_right_offset is not None
        assert self._command_center is not None
        assert self._rotation_input_left_ref is not None
        assert self._rotation_input_right_ref is not None

        left_translation = input_left[:3, 3] - self._input_left_lock[:3, 3]
        right_translation = input_right[:3, 3] - self._input_right_lock[:3, 3]
        desired_center = self._robot_center_lock + 0.5 * (
            left_translation + right_translation
        )

        left_rotation = input_left[:3, :3] @ self._rotation_input_left_ref.T
        right_rotation = input_right[:3, :3] @ self._rotation_input_right_ref.T
        disagreement = _rotation_vector(left_rotation @ right_rotation.T)
        self.last_rotation_disagreement_rad = float(np.linalg.norm(disagreement))
        desired_rotation = self._command_rotation
        if self.last_rotation_disagreement_rad > self.max_rotation_disagreement_rad:
            self._rotation_suspended = True
        elif self._rotation_suspended:
            # Drop rotation accumulated while the controllers disagreed. Rebase
            # at the recovered poses so disabling the hold cannot cause catch-up.
            self._rotation_input_left_ref = input_left[:3, :3].copy()
            self._rotation_input_right_ref = input_right[:3, :3].copy()
            self._rotation_command_ref = self._command_rotation.copy()
            self._rotation_suspended = False
        else:
            common_rotation_vector = 0.5 * (
                _rotation_vector(left_rotation) + _rotation_vector(right_rotation)
            )
            desired_rotation = (
                _rotation_from_vector(common_rotation_vector)
                @ self._rotation_command_ref
            )

        previous_update_s = self._last_update_s
        elapsed_s = 0.0 if previous_update_s is None else max(0.0, now_s - previous_update_s)
        elapsed_s = min(elapsed_s, 0.10)
        translation_step = _limit_vector(
            desired_center - self._command_center,
            None
            if self.max_translation_speed_m_s is None
            else self.max_translation_speed_m_s * elapsed_s,
        )
        candidate_center = self._command_center + translation_step

        rotation_error = desired_rotation @ self._command_rotation.T
        rotation_step = _limit_vector(
            _rotation_vector(rotation_error),
            None
            if self.max_rotation_speed_rad_s is None
            else self.max_rotation_speed_rad_s * elapsed_s,
        )
        candidate_rotation = _rotation_from_vector(rotation_step) @ self._command_rotation

        left_target, right_target = self._build_locked_pair(
            candidate_center, candidate_rotation
        )
        self.last_workspace_limited = bool(
            validator is not None and not validator(left_target, right_target)
        )
        if self.last_workspace_limited:
            left_target, right_target = self._build_locked_pair(
                self._command_center, self._command_rotation
            )
        else:
            self._command_center = candidate_center
            self._command_rotation = candidate_rotation
        self._last_update_s = now_s
        return left_target, right_target

    def _build_locked_pair(
        self,
        center: np.ndarray,
        rotation: np.ndarray,
    ) -> tuple[Pose, Pose]:
        assert self._robot_left_lock is not None
        assert self._robot_right_lock is not None
        assert self._robot_left_offset is not None
        assert self._robot_right_offset is not None
        left = self._robot_left_lock.copy()
        right = self._robot_right_lock.copy()
        left[:3, 3] = center + rotation @ self._robot_left_offset
        right[:3, 3] = center + rotation @ self._robot_right_offset
        left[:3, :3] = rotation @ self._robot_left_lock[:3, :3]
        right[:3, :3] = rotation @ self._robot_right_lock[:3, :3]
        return left, right

    def _free_targets(
        self,
        input_left: Pose,
        input_right: Pose,
        validator: Optional[TargetValidator],
    ) -> tuple[Pose, Pose]:
        if self._free_input_left_ref is None or self._free_input_right_ref is None:
            self.last_workspace_limited = False
            return input_left, input_right
        assert self._free_robot_left_ref is not None
        assert self._free_robot_right_ref is not None
        left = self._relative_target(
            input_left, self._free_input_left_ref, self._free_robot_left_ref
        )
        right = self._relative_target(
            input_right, self._free_input_right_ref, self._free_robot_right_ref
        )
        self.last_workspace_limited = bool(
            validator is not None and not validator(left, right)
        )
        if self.last_workspace_limited:
            assert self._free_last_left is not None
            assert self._free_last_right is not None
            return self._free_last_left.copy(), self._free_last_right.copy()
        self._free_last_left = left.copy()
        self._free_last_right = right.copy()
        return left, right

    @staticmethod
    def _relative_target(current: Pose, input_ref: Pose, robot_ref: Pose) -> Pose:
        target = robot_ref.copy()
        target[:3, 3] += current[:3, 3] - input_ref[:3, 3]
        target[:3, :3] = (
            current[:3, :3] @ input_ref[:3, :3].T @ robot_ref[:3, :3]
        )
        return target
