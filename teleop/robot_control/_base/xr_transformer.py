"""XR 数据变换器 — OpenXR 原始数据 → IK 就绪数据。"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any
import numpy as np


@dataclass
class XRProcessedData:
    """IK 就绪数据。"""
    left_wrist_pose: np.ndarray    # (4,4)
    right_wrist_pose: np.ndarray   # (4,4)
    head_kwargs: dict              # 传给 ik.solve_ik() 的 **kwargs
    left_hand_pos: np.ndarray | None = None
    right_hand_pos: np.ndarray | None = None


def fast_mat_inv(mat: np.ndarray) -> np.ndarray:
    ret = np.eye(4)
    ret[:3, :3] = mat[:3, :3].T
    ret[:3, 3] = -mat[:3, :3].T @ mat[:3, 3]
    return ret


def safe_mat_update(prev, mat):
    det = np.linalg.det(mat)
    if not np.isfinite(det) or np.isclose(det, 0.0, atol=1e-6):
        return prev, False
    return mat, True


def safe_rot_update(prev_rot_array, rot_array):
    dets = np.linalg.det(rot_array)
    if not np.all(np.isfinite(dets)) or np.any(np.isclose(dets, 0.0, atol=1e-6)):
        return prev_rot_array, False
    return rot_array, True


class XRTransformer(ABC):
    """OpenXR 原始数据 → IK 就绪数据的机器人相关变换器。

    子类必须定义：
      - T_ROBOT_OPENXR:  Basis 变换矩阵 (OpenXR → Robot)
      - T_LEFT_ARM:      左臂初始位姿变换
      - T_RIGHT_ARM:     右臂初始位姿变换
      - _transform_head():    头部数据 → IK kwargs
      - _preprocess_arm_poses():  臂部预处理
      - S_LEFT_HAND_TO_EE: 相似变换 (OpenXR 左手 → 机器人左末端)
      - S_RIGHT_HAND_TO_EE: 相似变换 (OpenXR 右手 → 机器人右末端)
    """

    def __init__(self, arm_ik, arm_scale: float = 1.0):
        self.arm_ik = arm_ik
        self.arm_scale = arm_scale

    @property
    @abstractmethod
    def T_ROBOT_OPENXR(self) -> np.ndarray: ...

    @property
    @abstractmethod
    def T_LEFT_ARM(self) -> np.ndarray: ...

    @property
    @abstractmethod
    def T_RIGHT_ARM(self) -> np.ndarray: ...

    # 控制器追踪路径的初始位姿变换（与 hand tracking 不同）
    @property
    @abstractmethod
    def T_LEFT_ARM_CTRL(self) -> np.ndarray: ...

    @property
    @abstractmethod
    def T_RIGHT_ARM_CTRL(self) -> np.ndarray: ...

    # 控制器追踪路径下 head 的额外变换（hand tracking 路径恒为 identity）
    @property
    def T_HEAD_CTRL(self) -> np.ndarray:
        return np.eye(4)

    # ── 相对位姿模式：相似变换矩阵（占位，待用户确认后填入） ───────────

    @property
    @abstractmethod
    def S_LEFT_HAND_TO_EE(self) -> np.ndarray:
        """相似变换: OpenXR 左手坐标系 → 机器人左末端坐标系。

        该矩阵应保持同一物理运动在两个坐标系下的等价性。
        TODO: 用户确认后填入正确的变换矩阵。
        """
        ...

    @property
    @abstractmethod
    def S_RIGHT_HAND_TO_EE(self) -> np.ndarray:
        """相似变换: OpenXR 右手坐标系 → 机器人右末端坐标系。"""
        ...

    @property
    @abstractmethod
    def S_LEFT_HAND_TO_RIGHT_EE(self) -> np.ndarray:
        """相似变换: OpenXR 左手坐标系 → 机器人右末端坐标系（镜像映射）。"""
        ...

    @property
    @abstractmethod
    def S_RIGHT_HAND_TO_LEFT_EE(self) -> np.ndarray:
        """相似变换: OpenXR 右手坐标系 → 机器人左末端坐标系（镜像映射）。"""
        ...

    # ── 相对位姿变换管线（与 transform() 完全独立） ──────────────────

    def transform_relative_pose(
        self,
        current_wrist_xr: np.ndarray,
        init_wrist_ref_xr: np.ndarray,
        reference_ee_pose: np.ndarray,
        arm_scale: float = 1.0,
    ) -> np.ndarray:
        """相对位姿变换 — XR 原始空间 → 机器人空间 → 叠加到冻结 EE 基准。

        流程：
          Step 1: ΔX = X_cur @ inv(X_ref)
                  手部增量，在 XR 世界坐标系下
          Step 2: ΔR_robot = R @ ΔR_xr @ R^T  (旋转：相似变换)
                  Δt_robot = R @ Δt_xr         (平移：左乘)
                  其中 R = T_ROBOT_OPENXR[:3,:3]
          Step 3: Δt_robot *= arm_scale   (可选平移缩放)
          Step 4: E_target = ΔH_robot @ E_ref
                  叠加到参考末端（左乘，增量作用于世界坐标系）

        参数:
            current_wrist_xr:   当前帧手腕位姿 (4×4), XR 原始坐标系 (OpenXR)
            init_wrist_ref_xr:  A键按下时冻结的手腕参考位姿 (4×4), XR 原始坐标系
            reference_ee_pose:  机器人参考末端位姿 (4×4), robot 约定
                               (激活瞬间的 FK 位姿，冻结)
            arm_scale: 手臂运动缩放因子 (默认 1.0，仅缩放平移)

        返回:
            E_target: (4×4) 目标末端位姿，robot 约定，直接作为 IK 输入
        """
        R = self.T_ROBOT_OPENXR[:3, :3]

        # Step 1: 物理位移（XR 世界坐标系下，与旋转完全解耦）
        displacement_xr = current_wrist_xr[:3, 3] - init_wrist_ref_xr[:3, 3]

        # Step 2: 旋转增量（XR 世界坐标系下）
        ΔR_xr = current_wrist_xr[:3, :3] @ init_wrist_ref_xr[:3, :3].T

        # Step 3: 基底变换到机器人坐标系
        ΔR_robot = R @ ΔR_xr @ R.T
        displacement_robot = R @ displacement_xr

        # Step 4: 可选平移缩放
        if arm_scale != 1.0:
            displacement_robot *= arm_scale

        # Step 5: 解耦叠加到参考末端
        E_target = reference_ee_pose.copy()
        E_target[:3, :3] = ΔR_robot @ reference_ee_pose[:3, :3]  # 旋转叠加
        E_target[:3, 3]  = reference_ee_pose[:3, 3] + displacement_robot  # 纯位移叠加

        return E_target

    @abstractmethod
    def _transform_head(self, tele_data, init_head_pose, head_is_valid,
                        head_pose: np.ndarray,
                        hand_tracking: bool = True) -> dict: ...

    @abstractmethod
    def _preprocess_arm_poses(self, left: np.ndarray, right: np.ndarray,
                              hand_tracking: bool = True
                              ) -> tuple[np.ndarray, np.ndarray]: ...

    def _xr_to_robot(self, pose_4x4: np.ndarray) -> np.ndarray:
        """Basis 变换: OpenXR → Robot。"""
        T = self.T_ROBOT_OPENXR
        return T @ pose_4x4 @ fast_mat_inv(T)

    def transform(self, tele_data, control_mode: str | None = None) -> XRProcessedData:
        """完整变换管线。

        Args:
            tele_data: XR 原始数据
            control_mode: 控制模式字符串 (如 "arms_only", "arms_head")。
                         仅当 control_mode == "arms_only" 时，手臂位置相对于实时头部；
                         其他模式（含头部/躯干）仍使用初始头部作为参考。
        """
        CONST_HEAD_POSE = np.array([[1,0,0,0],[0,1,0,1.5],[0,0,1,-0.2],[0,0,0,1]])

        # 追踪模式
        hand_tracking = getattr(tele_data, 'use_hand_tracking', True)

        head_pose_raw = getattr(tele_data, 'head_pose', CONST_HEAD_POSE)
        head_pose, head_is_valid = safe_mat_update(CONST_HEAD_POSE, head_pose_raw)
        left_raw = tele_data.left_wrist_pose.copy()
        right_raw = tele_data.right_wrist_pose.copy()

        # Basis 变换
        left_robot  = self._xr_to_robot(left_raw)
        right_robot = self._xr_to_robot(right_raw)

        # 初始位姿变换（hand 或 controller 路径）
        if hand_tracking:
            left_robot  = left_robot @ self.T_LEFT_ARM
            right_robot = right_robot @ self.T_RIGHT_ARM
        else:
            left_robot  = left_robot @ self.T_LEFT_ARM_CTRL
            right_robot = right_robot @ self.T_RIGHT_ARM_CTRL

        # ←── WORLD→HEAD 平移（手臂位置参考系）──
        # arms_only: 相对于实时头部（手臂跟随头部移动）
        # 其他模式：  相对于初始头部（保持世界坐标系固定）
        if control_mode == "arms_only":
            head_robot = self._xr_to_robot(head_pose)
            left_robot[:3, 3] -= head_robot[:3, 3]
            right_robot[:3, 3] -= head_robot[:3, 3]
        else:
            init_head_robot = getattr(self, '_init_head_robot', None)
            if init_head_robot is not None:
                left_robot[:3, 3] -= init_head_robot[:3, 3]
                right_robot[:3, 3] -= init_head_robot[:3, 3]

        # 臂部预处理
        left, right = self._preprocess_arm_poses(left_robot, right_robot, hand_tracking)

        # 头部处理
        init_head = getattr(tele_data, 'init_head_pose', None)
        head_kwargs = self._transform_head(tele_data, init_head, head_is_valid, head_pose, hand_tracking)

        return XRProcessedData(
            left_wrist_pose=left,
            right_wrist_pose=right,
            head_kwargs=head_kwargs,
        )
