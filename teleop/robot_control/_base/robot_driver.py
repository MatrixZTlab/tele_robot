from __future__ import annotations

from abc import ABC, abstractmethod
import time
from typing import Any, Optional

import numpy as np

from teleop.robot_control._base.control_mode import ControlMode
from teleop.robot_control._base.robot_config import RobotConfig
from teleop.robot_control._base.arm_ik import BaseArmIK, IKResult
from teleop.robot_control._base.arm_controller import BaseArmController
from teleop.robot_control._base.xr_transformer import XRTransformer, XRProcessedData
from teleop.robot_control.handler_registry import HandlerRegistry


class RobotDriver(ABC):
    """机器人遥操作驱动 — 封装完整的控制管线。

    主循环只需调用 step()，所有机器人差异内聚在此。
    """

    def __init__(self, config: RobotConfig, control_mode: ControlMode,
                 frequency: float = 50.0, simulation_mode: bool = True,
                 arm_scale: float = 1.0, verbose: bool = False):
        self.config = config
        self.control_mode = control_mode
        self.frequency = frequency
        self.simulation_mode = simulation_mode
        self.arm_scale = arm_scale
        self.verbose = verbose

        # 由子类在 _build_components() 中创建
        self.ik: Optional[BaseArmIK] = None
        self.controller: Optional[BaseArmController] = None
        self.xr_transformer: Optional[XRTransformer] = None
        self.handler_registry = HandlerRegistry()

        # 相对位姿模式：激活瞬间的 EE 位姿（冻结基准，防漂移）
        self._init_left_ee: Optional[np.ndarray] = None
        self._init_right_ee: Optional[np.ndarray] = None
        # 回放时上一次 gripper 指令（未下发新指令时保持上一状态）
        self._last_ee_action: Optional[list] = None
        self.last_step_timing_ms: dict[str, float] = {}
        self.last_step_alignment_ns: dict[str, int] = {}

        # 子类构建
        self._build_components()

    # ── 子类必须覆写 ────────────────────────────────────────

    @abstractmethod
    def _build_components(self) -> None:
        """创建 self.ik, self.controller 实例。"""
        ...

    # ── 子类可选的预处理钩子 ────────────────────────────────

    def _dispatch_commands(self, ik_result: IKResult) -> None:
        """将 IK 结果分发到 controller。

        默认只下发双臂。子类可覆写增加头部/躯干控制。
        """
        self.controller.servo_dual_arm(ik_result.arm_q, ik_result.arm_tau)

    def _collect_recording_state(self, ik_result: IKResult,
                                 current_q: np.ndarray) -> dict:
        """收集本帧录制数据。子类可覆写。"""
        return {
            "left_arm_state":  current_q[:7].tolist(),
            "right_arm_state": current_q[-7:].tolist(),
            "left_arm_action": ik_result.arm_q[:7].tolist(),
            "right_arm_action": ik_result.arm_q[-7:].tolist(),
            **self.handler_registry.collect_recording_state(),
        }

    def _on_step_done(self, ik_result: IKResult, tele_data: Any) -> None:
        """每帧 step 执行完毕后的回调。子类可选覆写。"""
        pass

    # ── 公有方法 ─────────────────────────────────────────────

    def _get_current_ee_poses(self, current_q: np.ndarray
                               ) -> tuple[np.ndarray, np.ndarray]:
        """获取机器人当前末端位姿 (4×4 homogeneous)，来自 FK。

        current_q 为弧度值（controller 输出），内部转换为度后计算 FK。
        返回:
            tuple: (left_ee_pose, right_ee_pose) 各为 (4,4)，robot 约定。
        """
        if self.ik is not None:
            # FK 方法期望度值，但 current_q 是弧度
            q_deg = np.rad2deg(np.asarray(current_q, dtype=float))
            return self.ik.get_dual_arm_ee_poses(q_deg)
        # fallback: identity
        return np.eye(4), np.eye(4)

    def step(self, tele_data: Any) -> dict:
        """执行一帧遥操作管线，返回录制数据快照。

        流程：状态获取 → XR 变换 → IK 求解 → 命令分发 → 录制收集
        根据 tele_data.teleop_mode 在 relative_head 和 relative_pose 模式间分支。
        """
        step_start = time.perf_counter_ns()
        # ① 状态
        if hasattr(self.controller, 'get_current_dual_arm_state'):
            current_q, current_dq, state_receive_ns = (
                self.controller.get_current_dual_arm_state()
            )
        else:
            current_q = self.controller.get_current_dual_arm_q()
            current_dq = self.controller.get_current_dual_arm_dq()
            state_receive_ns = 0
        state_done = time.perf_counter_ns()

        teleop_mode = getattr(tele_data, 'teleop_mode', 'relative_head')

        if teleop_mode == "relative_head":
            # ── 现有管线 ──
            xr = self.xr_transformer.transform(tele_data, self.control_mode.value)
            transform_done = time.perf_counter_ns()
            ik_result = self.ik.solve_ik(
                xr.left_wrist_pose, xr.right_wrist_pose,
                current_q, current_dq,
                **xr.head_kwargs,
            )

        elif teleop_mode == "relative_pose":
            # ── 相对位姿管线：FK → 相似变换 → IK ──
            left_ee, right_ee = self._get_current_ee_poses(current_q)

            # ── 镜像映射：左手数据 → 右臂，右手数据 → 左臂 ──
            l_ref = getattr(tele_data, 'left_wrist_ref', None)
            if l_ref is not None:
                if self._init_right_ee is None:   # 首次或重新激活时捕获
                    self._init_right_ee = right_ee.copy()
                    print(f"[REL_POSE] init_right_ee captured from FK (pos={right_ee[:3,3]})")
                right_target = self.xr_transformer.transform_relative_pose(
                    tele_data.left_wrist_pose, l_ref,
                    self._init_right_ee,
                    arm_scale=self.arm_scale,
                )
            else:
                right_target = right_ee
                if self._init_right_ee is not None:
                    print("[REL_POSE] left_wrist_ref cleared, init_right_ee reset")
                self._init_right_ee = None

            r_ref = getattr(tele_data, 'right_wrist_ref', None)
            if r_ref is not None:
                if self._init_left_ee is None:   # 首次或重新激活时捕获
                    self._init_left_ee = left_ee.copy()
                    print(f"[REL_POSE] init_left_ee captured from FK (pos={left_ee[:3,3]})")
                left_target = self.xr_transformer.transform_relative_pose(
                    tele_data.right_wrist_pose, r_ref,
                    self._init_left_ee,
                    arm_scale=self.arm_scale,
                )
            else:
                left_target = left_ee
                if self._init_left_ee is not None:
                    print("[REL_POSE] right_wrist_ref cleared, init_left_ee reset")
                self._init_left_ee = None

            # relative_pose 模式不做头部 IK
            transform_done = time.perf_counter_ns()
            ik_result = self.ik.solve_ik(
                left_target, right_target,
                current_q, current_dq,
                head_target=None,
            )

            # ── 未激活臂绕过 IK + 滤波，直接用当前关节角保持原位 ──
            if r_ref is None:
                ik_result.arm_q[:7] = current_q[:7]
                ik_result.arm_tau[:7] = np.zeros(7)
            if l_ref is None:
                ik_result.arm_q[7:14] = current_q[7:14]
                ik_result.arm_tau[7:14] = np.zeros(7)

        else:
            # fallback: relative_head
            xr = self.xr_transformer.transform(tele_data, self.control_mode.value)
            transform_done = time.perf_counter_ns()
            ik_result = self.ik.solve_ik(
                xr.left_wrist_pose, xr.right_wrist_pose,
                current_q, current_dq,
                **xr.head_kwargs,
            )
        ik_done = time.perf_counter_ns()

        # ④ 命令分发
        self._dispatch_commands(ik_result)
        dispatch_done = time.perf_counter_ns()

        # ⑤ 收尾回调
        self._on_step_done(ik_result, tele_data)

        # ⑦ 录制快照
        snapshot = self._collect_recording_state(ik_result, current_q)
        collect_done = time.perf_counter_ns()
        self.last_step_timing_ms = {
            "state": (state_done - step_start) / 1e6,
            "transform": (transform_done - state_done) / 1e6,
            "ik": (ik_done - transform_done) / 1e6,
            "dispatch": (dispatch_done - ik_done) / 1e6,
            "collect": (collect_done - dispatch_done) / 1e6,
            "total": (collect_done - step_start) / 1e6,
        }
        self.last_step_alignment_ns = {
            "state_receive_monotonic": int(state_receive_ns or 0),
        }
        return snapshot

    def reset_relative_pose_state(self, side: str = 'both') -> None:
        """当外层激活臂或切换模式时调用，强制清除 _init_ee 以便 step() 重新捕获 FK 基准。

        Args:
            side: 'left'  → 清除 _init_left_ee（右手→左臂）
                  'right' → 清除 _init_right_ee（左手→右臂）
                  'both'  → 全部清除
        """
        if side in ('left', 'both'):
            self._init_left_ee = None
        if side in ('right', 'both'):
            self._init_right_ee = None

    def go_home(self, timeout: float = 10.0) -> None:
        """回到零位。"""
        if self.controller is not None:
            self.controller.go_home(timeout=timeout)

    def speed_gradual_max(self, t: float = 5.0) -> None:
        if self.controller is not None:
            self.controller.speed_gradual_max(t)

    def speed_instant_max(self) -> None:
        if self.controller is not None:
            self.controller.speed_instant_max()

    def replay_movej(self, joints: list[float], duration: float) -> None:
        """回放用的 MoveJ。"""
        if self.controller is not None:
            self.controller.move_joints_timed(joints, duration)

    # ── 回放 ────────────────────────────────────────────────

    def replay_point(self, point: dict,
                     control_mode: "ControlMode") -> None:
        """回放单个轨迹点 — 子类可覆写以选择最优路径。

        point 格式: {"type": "MoveJ"|"ServoJ", "joint_pose": [...°],
                     "duration": 0.5, "head_pose": [yaw°, pitch°]}
        """
        import numpy as np
        from teleop.robot_control._base.control_mode import ControlMode

        point_type = point.get("type")
        joint_pose_deg = point.get("joint_pose")
        if joint_pose_deg is None:
            return

        joint_pose_rad = np.deg2rad(
            np.asarray(joint_pose_deg, dtype=float)
        ).tolist()

        # 按 control_mode 截取/判断维度
        dim_needed = control_mode.ik_dof
        if len(joint_pose_rad) < dim_needed:
            raise ValueError(
                f"轨迹 joint_pose 维度 ({len(joint_pose_rad)}) "
                f"小于 control_mode 需要 ({dim_needed})"
            )

        # 手臂部分 (前 14 维)
        arm_rad = joint_pose_rad[:14]

        # ── 末端执行器（吸盘/夹爪）：与手臂命令同步下发 ──
        ee_action = point.get("ee_action")
        if ee_action is not None and len(ee_action) >= 2:
            self._last_ee_action = list(ee_action[:2])
        if self._last_ee_action is not None:
            for arm_idx, val in enumerate(self._last_ee_action):
                self.controller.set_ee_gripper(arm_idx, float(val))

        if point_type == "MoveJ":
            duration = float(point.get("duration", 0.5))
            if duration <= 0:
                duration = 0.5
            self.controller.move_joints_timed_and_verify(arm_rad, duration)
        else:
            self.controller.servo_dual_arm(arm_rad, np.zeros(14))

        # 头部
        if (control_mode.solve_head
                and len(joint_pose_rad) >= 16):
            head_q = joint_pose_rad[14:16]
            self.controller.ctrl_head(head_q)
        elif point.get("head_pose") is not None:
            head_deg = point["head_pose"]
            head_rad = np.deg2rad(np.asarray(head_deg, dtype=float))[:2]
            self.controller.ctrl_head(head_rad.tolist())
