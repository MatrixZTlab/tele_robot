"""Experimental VR-relative teleoperation and MuJoCo LM IK for TOPSTAR_H1.

This module is intentionally isolated from the live teleoperation pipeline.  It
ports the fixed-reference wrist tracking behavior from ``vr_teleop`` to the
matrix representation used by TeleVuer, and provides an optional MuJoCo-backed
dual-arm IK solver for offline evaluation.
"""

from __future__ import annotations

import importlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


_POSE_SHAPE = (4, 4)
_ROTATION_SHAPE = (3, 3)
_H1_ARM_DOF = 14

H1_XR_TO_ROBOT_ROTATION = np.array(
    [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    dtype=np.float64,
)
H1_LEFT_ARM_JOINT_NAMES = (
    "Robot_Left_Hand_base_Joint",
    "Robot_Left_Hand_1_Joint",
    "Robot_Left_Hand_2_Joint",
    "Robot_Left_Hand_3_Joint",
    "Robot_Left_Hand_4_Joint",
    "Robot_Left_Hand_5_Joint",
    "Robot_Left_Hand_6_Joint",
)
H1_RIGHT_ARM_JOINT_NAMES = tuple(
    name.replace("Left", "Right") for name in H1_LEFT_ARM_JOINT_NAMES
)


def _validate_rotation(rotation: np.ndarray, name: str) -> np.ndarray:
    value = np.asarray(rotation, dtype=np.float64)
    if value.shape != _ROTATION_SHAPE:
        raise ValueError(f"{name} must have shape (3, 3), got {value.shape}")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must contain only finite values")
    if not np.allclose(value.T @ value, np.eye(3), atol=1e-5):
        raise ValueError(f"{name} must be an orthonormal rotation matrix")
    if not np.isclose(np.linalg.det(value), 1.0, atol=1e-5):
        raise ValueError(f"{name} must have determinant +1")
    return value.copy()


def _validate_pose(pose: np.ndarray, name: str) -> np.ndarray:
    value = np.asarray(pose, dtype=np.float64)
    if value.shape != _POSE_SHAPE:
        raise ValueError(f"{name} must have shape (4, 4), got {value.shape}")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must contain only finite values")
    if not np.allclose(value[3], [0.0, 0.0, 0.0, 1.0], atol=1e-7):
        raise ValueError(f"{name} must have homogeneous last row [0, 0, 0, 1]")
    _validate_rotation(value[:3, :3], f"{name} rotation")
    return value.copy()


def _rotation_angle(rotation: np.ndarray) -> float:
    value = _validate_rotation(rotation, "rotation")
    cosine = np.clip((np.trace(value) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.arccos(cosine))


class VRRelativePoseTracker:
    """Map a wrist pose residual onto a captured H1 end-effector pose.

    The first valid wrist sample after construction or :meth:`reset` becomes
    the fixed wrist reference.  Every later target is recomputed relative to
    that reference; adjacent-frame deltas are never integrated.
    """

    def __init__(
        self,
        initial_ee_pose: np.ndarray,
        xr_to_robot_rotation: np.ndarray,
        *,
        position_scale: float = 1.5,
        ema_alpha: float = 0.8,
        position_deadband: float = 0.0,
        rotation_deadband_deg: float = 0.0,
    ) -> None:
        if not np.isfinite(position_scale) or position_scale <= 0.0:
            raise ValueError("position_scale must be finite and greater than zero")
        if not np.isfinite(ema_alpha) or not 0.0 < ema_alpha <= 1.0:
            raise ValueError("ema_alpha must be finite and in (0, 1]")
        if not np.isfinite(position_deadband) or position_deadband < 0.0:
            raise ValueError("position_deadband must be finite and non-negative")
        if not np.isfinite(rotation_deadband_deg) or rotation_deadband_deg < 0.0:
            raise ValueError(
                "rotation_deadband_deg must be finite and non-negative"
            )

        self._basis = _validate_rotation(
            xr_to_robot_rotation, "xr_to_robot_rotation"
        )
        self._position_scale = float(position_scale)
        self._ema_alpha = float(ema_alpha)
        self._position_deadband = float(position_deadband)
        self._rotation_deadband_rad = math.radians(rotation_deadband_deg)

        self._initial_ee_pose = _validate_pose(initial_ee_pose, "initial_ee_pose")
        self._target_pose = self._initial_ee_pose.copy()
        self._reference_wrist_pose: np.ndarray | None = None
        self._smoothed_translation: np.ndarray | None = None
        self._rotation_residual: np.ndarray | None = None

    @property
    def initialized(self) -> bool:
        return self._reference_wrist_pose is not None

    @property
    def target_pose(self) -> np.ndarray:
        return self._target_pose.copy()

    @property
    def translation_residual(self) -> np.ndarray | None:
        if self._smoothed_translation is None:
            return None
        return self._smoothed_translation.copy()

    @property
    def rotation_residual(self) -> np.ndarray | None:
        if self._rotation_residual is None:
            return None
        return self._rotation_residual.copy()

    def reset(self, initial_ee_pose: np.ndarray | None = None) -> None:
        """Clear the wrist reference and optionally capture a new H1 EE pose."""
        if initial_ee_pose is not None:
            self._initial_ee_pose = _validate_pose(
                initial_ee_pose, "initial_ee_pose"
            )
        self._target_pose = self._initial_ee_pose.copy()
        self._reference_wrist_pose = None
        self._smoothed_translation = None
        self._rotation_residual = None

    def update(self, wrist_pose_xr: np.ndarray) -> np.ndarray:
        """Update and return the target H1 end-effector pose."""
        wrist = _validate_pose(wrist_pose_xr, "wrist_pose_xr")
        if self._reference_wrist_pose is None:
            self._reference_wrist_pose = wrist
            return self.target_pose

        translation = self._basis @ (
            wrist[:3, 3] - self._reference_wrist_pose[:3, 3]
        )
        if self._smoothed_translation is None:
            self._smoothed_translation = translation
        else:
            self._smoothed_translation = (
                self._ema_alpha * translation
                + (1.0 - self._ema_alpha) * self._smoothed_translation
            )
        if np.linalg.norm(self._smoothed_translation) < self._position_deadband:
            self._smoothed_translation = np.zeros(3, dtype=np.float64)

        rotation_xr = (
            wrist[:3, :3] @ self._reference_wrist_pose[:3, :3].T
        )
        rotation_robot = self._basis @ rotation_xr @ self._basis.T
        if _rotation_angle(rotation_robot) < self._rotation_deadband_rad:
            rotation_robot = np.eye(3)
        self._rotation_residual = rotation_robot

        target = self._initial_ee_pose.copy()
        target[:3, 3] += self._position_scale * self._smoothed_translation
        target[:3, :3] = rotation_robot @ self._initial_ee_pose[:3, :3]
        self._target_pose = target
        return self.target_pose


@dataclass(frozen=True)
class LMIKResult:
    """Result of one dual-arm MuJoCo LM solve."""

    arm_q: np.ndarray
    converged: bool
    iterations: int
    translation_error_norm: float
    rotation_error_norm: float

    def __post_init__(self) -> None:
        arm_q = np.asarray(self.arm_q, dtype=np.float64).copy()
        if arm_q.shape != (_H1_ARM_DOF,):
            raise ValueError(f"arm_q must contain {_H1_ARM_DOF} values")
        if not np.all(np.isfinite(arm_q)):
            raise ValueError("arm_q must contain only finite values")
        arm_q.setflags(write=False)
        object.__setattr__(self, "arm_q", arm_q)


def _rotation_error(target_rotation: np.ndarray, current_rotation: np.ndarray) -> np.ndarray:
    """Return the shortest SO(3) axis-angle error vector."""
    target = _validate_rotation(target_rotation, "target_rotation")
    current = _validate_rotation(current_rotation, "current_rotation")
    relative = target @ current.T
    cosine = np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
    angle = float(np.arccos(cosine))
    if angle < 1e-8:
        return np.zeros(3, dtype=np.float64)

    if math.pi - angle < 1e-6:
        eigenvalues, eigenvectors = np.linalg.eig(relative)
        index = int(np.argmin(np.abs(eigenvalues - 1.0)))
        axis = np.real(eigenvectors[:, index])
        norm = np.linalg.norm(axis)
        if norm < 1e-12:
            raise ValueError("could not determine rotation axis near pi")
        axis /= norm
        return axis * angle

    axis = np.array(
        [
            relative[2, 1] - relative[1, 2],
            relative[0, 2] - relative[2, 0],
            relative[1, 0] - relative[0, 1],
        ],
        dtype=np.float64,
    )
    axis /= 2.0 * math.sin(angle)
    return axis * angle


def _stack_dual_arm_system(
    left_error: np.ndarray,
    right_error: np.ndarray,
    left_jacobian: np.ndarray,
    right_jacobian: np.ndarray,
    rotation_weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Stack left/right six-dimensional errors and Jacobians."""
    if not np.isfinite(rotation_weight) or rotation_weight < 0.0:
        raise ValueError("rotation_weight must be finite and non-negative")
    left_e = np.asarray(left_error, dtype=np.float64)
    right_e = np.asarray(right_error, dtype=np.float64)
    left_j = np.asarray(left_jacobian, dtype=np.float64)
    right_j = np.asarray(right_jacobian, dtype=np.float64)
    if left_e.shape != (6,) or right_e.shape != (6,):
        raise ValueError("left_error and right_error must each have shape (6,)")
    if left_j.ndim != 2 or left_j.shape[0] != 6:
        raise ValueError("left_jacobian must have shape (6, N)")
    if right_j.shape != left_j.shape:
        raise ValueError("right_jacobian must match left_jacobian shape")
    if not all(
        np.all(np.isfinite(value)) for value in (left_e, right_e, left_j, right_j)
    ):
        raise ValueError("errors and Jacobians must contain only finite values")

    error = np.concatenate(
        [
            left_e[:3],
            rotation_weight * left_e[3:],
            right_e[:3],
            rotation_weight * right_e[3:],
        ]
    )
    jacobian = np.vstack(
        [
            left_j[:3],
            rotation_weight * left_j[3:],
            right_j[:3],
            rotation_weight * right_j[3:],
        ]
    )
    return error, jacobian


def _damped_least_squares_step(
    jacobian: np.ndarray, error: np.ndarray, damping: float
) -> np.ndarray:
    """Compute one damped LM/DLS joint-space update."""
    jac = np.asarray(jacobian, dtype=np.float64)
    err = np.asarray(error, dtype=np.float64)
    if jac.ndim != 2:
        raise ValueError("jacobian must be a two-dimensional array")
    if err.shape != (jac.shape[0],):
        raise ValueError("error length must match jacobian rows")
    if not np.all(np.isfinite(jac)) or not np.all(np.isfinite(err)):
        raise ValueError("jacobian and error must contain only finite values")
    if not np.isfinite(damping) or damping <= 0.0:
        raise ValueError("damping must be finite and greater than zero")
    regularized = jac @ jac.T + damping * np.eye(jac.shape[0])
    return jac.T @ np.linalg.solve(regularized, err)


def _load_mujoco() -> Any:
    """Load the optional MuJoCo dependency only when the IK backend is used."""
    try:
        return importlib.import_module("mujoco")
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "MuJoCo IK requires the optional 'mujoco' package; "
            "install it with 'python -m pip install mujoco'"
        ) from exc


class H1MuJoCoLMIK:
    """Dual-arm damped LM IK using MuJoCo FK and tool-point Jacobians."""

    def __init__(
        self,
        mujoco_api: Any,
        model: Any,
        data: Any,
        left_joint_ids: np.ndarray,
        right_joint_ids: np.ndarray,
        left_body_id: int,
        right_body_id: int,
        ee_offset: Sequence[float],
    ) -> None:
        self._mj = mujoco_api
        self._model = model
        self._data = data
        self._joint_ids = np.concatenate(
            [
                np.asarray(left_joint_ids, dtype=np.int64),
                np.asarray(right_joint_ids, dtype=np.int64),
            ]
        )
        if self._joint_ids.shape != (_H1_ARM_DOF,):
            raise ValueError("H1MuJoCoLMIK requires seven left and seven right joints")
        self._qpos_addresses = np.asarray(
            [model.jnt_qposadr[joint_id] for joint_id in self._joint_ids],
            dtype=np.int64,
        )
        self._dof_addresses = np.asarray(
            [model.jnt_dofadr[joint_id] for joint_id in self._joint_ids],
            dtype=np.int64,
        )
        self._left_body_id = int(left_body_id)
        self._right_body_id = int(right_body_id)
        offset = np.asarray(ee_offset, dtype=np.float64)
        if offset.shape != (3,) or not np.all(np.isfinite(offset)):
            raise ValueError("ee_offset must contain three finite values")
        self._ee_offset = offset.copy()
        self._base_qpos = np.asarray(model.qpos0, dtype=np.float64).copy()

    @classmethod
    def from_h1_assets(cls, repo_root: str | Path | None = None) -> "H1MuJoCoLMIK":
        """Load the repository H1 URDF with its standard arm/body mapping."""
        root = (
            Path(repo_root).expanduser().resolve()
            if repo_root is not None
            else Path(__file__).resolve().parents[2]
        )
        return cls.from_urdf(
            root / "assets" / "topstar_h1" / "_tmp_h1_mujoco.urdf",
            H1_LEFT_ARM_JOINT_NAMES,
            H1_RIGHT_ARM_JOINT_NAMES,
            left_ee_body="Robot_Left_Hand_6_Link",
            right_ee_body="Robot_Right_Hand_6_Link",
            ee_offset=(0.0, 0.0, 0.03),
        )

    @classmethod
    def from_urdf(
        cls,
        urdf_path: str | Path,
        left_joint_names: Sequence[str],
        right_joint_names: Sequence[str],
        *,
        left_ee_body: str,
        right_ee_body: str,
        ee_offset: Sequence[float] = (0.0, 0.0, 0.03),
    ) -> "H1MuJoCoLMIK":
        """Load an H1 URDF and resolve the exact 14 arm degrees of freedom."""
        if len(left_joint_names) != 7 or len(right_joint_names) != 7:
            raise ValueError("H1 requires seven left and seven right joint names")
        path = Path(urdf_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"H1 URDF not found: {path}")
        mj = _load_mujoco()
        model = mj.MjModel.from_xml_path(str(path))
        data = mj.MjData(model)

        joint_ids = []
        for name in [*left_joint_names, *right_joint_names]:
            joint_id = int(mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, name))
            if joint_id < 0:
                raise ValueError(f"MuJoCo model does not contain H1 joint {name!r}")
            joint_ids.append(joint_id)
        left_body_id = int(
            mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, left_ee_body)
        )
        right_body_id = int(
            mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, right_ee_body)
        )
        if left_body_id < 0:
            raise ValueError(
                f"MuJoCo model does not contain left EE body {left_ee_body!r}"
            )
        if right_body_id < 0:
            raise ValueError(
                f"MuJoCo model does not contain right EE body {right_ee_body!r}"
            )
        return cls(
            mj,
            model,
            data,
            np.asarray(joint_ids[:7]),
            np.asarray(joint_ids[7:]),
            left_body_id,
            right_body_id,
            ee_offset,
        )

    def _validate_arm_q(self, arm_q: np.ndarray) -> np.ndarray:
        value = np.asarray(arm_q, dtype=np.float64)
        if value.shape != (_H1_ARM_DOF,):
            raise ValueError(f"current_arm_q must contain {_H1_ARM_DOF} values")
        if not np.all(np.isfinite(value)):
            raise ValueError("current_arm_q must contain only finite values")
        return value.copy()

    def _full_qpos(self, arm_q: np.ndarray) -> np.ndarray:
        qpos = self._base_qpos.copy()
        qpos[self._qpos_addresses] = arm_q
        return qpos

    def _tool_pose_and_jacobian(
        self, body_id: int
    ) -> tuple[np.ndarray, np.ndarray]:
        rotation = self._data.xmat[body_id].reshape(3, 3).copy()
        point = self._data.xpos[body_id].copy() + rotation @ self._ee_offset
        jacobian_position = np.zeros((3, self._model.nv), dtype=np.float64)
        jacobian_rotation = np.zeros((3, self._model.nv), dtype=np.float64)
        self._mj.mj_jac(
            self._model,
            self._data,
            jacobian_position,
            jacobian_rotation,
            point,
            body_id,
        )
        pose = np.eye(4)
        pose[:3, :3] = rotation
        pose[:3, 3] = point
        jacobian = np.vstack([jacobian_position, jacobian_rotation])
        return pose, jacobian[:, self._dof_addresses]

    def _evaluate(
        self, qpos: np.ndarray, left_target: np.ndarray, right_target: np.ndarray
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        float,
        float,
        np.ndarray,
        np.ndarray,
    ]:
        self._data.qpos[:] = qpos
        self._data.qvel[:] = 0.0
        self._mj.mj_forward(self._model, self._data)
        left_pose, left_jacobian = self._tool_pose_and_jacobian(self._left_body_id)
        right_pose, right_jacobian = self._tool_pose_and_jacobian(
            self._right_body_id
        )
        left_error = np.concatenate(
            [
                left_target[:3, 3] - left_pose[:3, 3],
                _rotation_error(left_target[:3, :3], left_pose[:3, :3]),
            ]
        )
        right_error = np.concatenate(
            [
                right_target[:3, 3] - right_pose[:3, 3],
                _rotation_error(right_target[:3, :3], right_pose[:3, :3]),
            ]
        )
        translation_norm = float(
            np.linalg.norm(np.concatenate([left_error[:3], right_error[:3]]))
        )
        rotation_norm = float(
            np.linalg.norm(np.concatenate([left_error[3:], right_error[3:]]))
        )
        return (
            left_error,
            right_error,
            translation_norm,
            rotation_norm,
            left_jacobian,
            right_jacobian,
        )

    def _clip_arm_joints(self, qpos: np.ndarray) -> None:
        for joint_id, qpos_address in zip(self._joint_ids, self._qpos_addresses):
            if bool(self._model.jnt_limited[joint_id]):
                lower, upper = self._model.jnt_range[joint_id]
                qpos[qpos_address] = np.clip(qpos[qpos_address], lower, upper)

    def forward_kinematics(
        self, current_arm_q: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return current left/right H1 tool poses for a 14-value arm state."""
        arm_q = self._validate_arm_q(current_arm_q)
        qpos = self._full_qpos(arm_q)
        self._data.qpos[:] = qpos
        self._data.qvel[:] = 0.0
        self._mj.mj_forward(self._model, self._data)
        left_pose, _ = self._tool_pose_and_jacobian(self._left_body_id)
        right_pose, _ = self._tool_pose_and_jacobian(self._right_body_id)
        return left_pose, right_pose

    def solve(
        self,
        left_target_pose: np.ndarray,
        right_target_pose: np.ndarray,
        current_arm_q: np.ndarray,
        *,
        max_iters: int = 30,
        tolerance: float = 1e-4,
        rotation_weight: float = 1.0,
        damping: float = 1e-3,
    ) -> LMIKResult:
        """Solve stacked left/right H1 tool targets from the measured arm state."""
        arm_q = self._validate_arm_q(current_arm_q)
        if not isinstance(max_iters, int) or max_iters <= 0:
            raise ValueError("max_iters must be a positive integer")
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("tolerance must be finite and greater than zero")
        if not np.isfinite(rotation_weight) or rotation_weight < 0.0:
            raise ValueError("rotation_weight must be finite and non-negative")
        if not np.isfinite(damping) or damping <= 0.0:
            raise ValueError("damping must be finite and greater than zero")
        left_target = _validate_pose(left_target_pose, "left_target_pose")
        right_target = _validate_pose(right_target_pose, "right_target_pose")

        qpos = self._full_qpos(arm_q)
        converged = False
        iterations = 0
        translation_norm = math.inf
        rotation_norm = math.inf
        for iteration in range(1, max_iters + 1):
            (
                left_error,
                right_error,
                translation_norm,
                rotation_norm,
                left_jacobian,
                right_jacobian,
            ) = self._evaluate(qpos, left_target, right_target)
            weighted_error, jacobian = _stack_dual_arm_system(
                left_error,
                right_error,
                left_jacobian,
                right_jacobian,
                rotation_weight,
            )
            if np.linalg.norm(weighted_error) < tolerance:
                converged = True
                iterations = iteration - 1
                break

            active_dq = _damped_least_squares_step(
                jacobian, weighted_error, damping
            )
            full_dq = np.zeros(self._model.nv, dtype=np.float64)
            full_dq[self._dof_addresses] = active_dq
            self._mj.mj_integratePos(self._model, qpos, full_dq, 1.0)
            self._clip_arm_joints(qpos)
            iterations = iteration

        if not converged:
            (
                left_error,
                right_error,
                translation_norm,
                rotation_norm,
                left_jacobian,
                right_jacobian,
            ) = self._evaluate(qpos, left_target, right_target)
            weighted_error, _ = _stack_dual_arm_system(
                left_error,
                right_error,
                left_jacobian,
                right_jacobian,
                rotation_weight,
            )
            converged = bool(np.linalg.norm(weighted_error) < tolerance)

        return LMIKResult(
            arm_q=qpos[self._qpos_addresses],
            converged=converged,
            iterations=iterations,
            translation_error_norm=translation_norm,
            rotation_error_norm=rotation_norm,
        )


__all__ = [
    "H1_LEFT_ARM_JOINT_NAMES",
    "H1_RIGHT_ARM_JOINT_NAMES",
    "H1_XR_TO_ROBOT_ROTATION",
    "H1MuJoCoLMIK",
    "LMIKResult",
    "VRRelativePoseTracker",
]
