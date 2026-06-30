from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np


class BaseArmController(ABC):
    """机械臂控制抽象接口。

    子类实现具体的通信协议（ROS2 / DDS）和硬件适配。
    """

    def __init__(self, config: "RobotConfig", control_mode: "ControlMode",
                 frequency: float = 50.0, simulation_mode: bool = True):
        from teleop.robot_control._base.robot_config import RobotConfig
        from teleop.robot_control._base.control_mode import ControlMode
        self.config: RobotConfig = config
        self.control_mode: ControlMode = control_mode
        self.frequency = frequency
        self.simulation_mode = simulation_mode
        self._motor_count: int = 0

    # ── 手臂 ────────────────────────────────────────────────────

    @abstractmethod
    def servo_dual_arm(self, q_target: np.ndarray,
                       tauff: Optional[np.ndarray] = None) -> None:
        """设置双臂伺服目标 (rad, sim convention)。"""
        ...

    @abstractmethod
    def get_current_dual_arm_q(self) -> np.ndarray:
        """获取当前双臂关节角 (rad, shape=(14,))。"""
        ...

    @abstractmethod
    def get_current_dual_arm_dq(self) -> np.ndarray:
        """获取当前双臂关节速度 (rad/s)。"""
        ...

    # ── 头部 ────────────────────────────────────────────────────

    def ctrl_head(self, head_input: Any) -> None:
        """控制头部。

        子类覆写以实现 IK 模式（接收 head_q）或 Euler 模式（接收 yaw/pitch）。
        """
        raise NotImplementedError(
            f"{self.config.model_name} does not support ctrl_head()"
        )

    def get_head_q(self) -> np.ndarray:
        """获取当前头部关节角。默认返回零。"""
        return np.zeros(2)

    # ── 躯干（预留）──────────────────────────────────────────────

    def ctrl_torso(self, torso_q: np.ndarray) -> None:
        """控制躯干 [lift, pitch]。"""
        raise NotImplementedError(f"Torso control not available for {self.config.model_name}")

    # ── sim → hw 统一转换 ──────────────────────────────────

    def _sim_to_hw_full_body(self, arm_q_sim_14, count: int = None):
        """14 关节 sim 值 → 全电机 hw 数组。

        手臂从 sim 转换后写入对应槽位；其余关节回读当前电机位置。
        count=None 时使用 _motor_count。
        """
        if count is None:
            count = self._motor_count
        arm_hw = self._sim_to_hw_arm(arm_q_sim_14)
        targets = [0.0] * count
        state = self._ros_node.state_buffer.get()
        if state is not None:
            n = min(count, len(state.motor_state))
            for idx in range(n):
                targets[idx] = float(state.motor_state[idx].q)
        for slot, v in zip(self.left_slots, arm_hw[:7]):
            targets[slot] = v
        for slot, v in zip(self.right_slots, arm_hw[7:]):
            targets[slot] = v
        return targets

    # ── MoveJ ───────────────────────────────────────────────────

    def move_joints_timed(self, joints: list[float],
                          duration: float) -> None:
        """带时长约束的关节空间运动。"""
        raise NotImplementedError(f"MoveJ not available for {self.config.model_name}")

    def go_home(self, timeout: float = 10.0) -> bool:
        """回到零位。"""
        raise NotImplementedError(f"go_home not available for {self.config.model_name}")

    def wait_for_hold_expire(self, timeout: float = 10.0) -> None:
        """阻塞直到 MoveJ 完成。"""
        pass

    def move_joints_timed_and_verify(self, joints: list[float],
                                      duration: float,
                                      tolerance: float = 0.05,
                                      extend_step: float = 2.0,
                                      max_total_time: float = 60.0) -> None:
        """MoveJ 并验证机器人是否到达目标，未到达则延长 duration。

        流程:
          1. 用初始 duration 下发 MoveJ
          2. 等待 duration 时间
          3. 检查当前关节角与目标的误差
          4. 误差 > tolerance 时，以 extend_step 为时长继续下发同样的目标
          5. 重复直到误差达标或超过 max_total_time

        Args:
            joints: 目标关节角 (rad)
            duration: 初始期望时长
            tolerance: 关节角容差 (rad), 默认 0.05 ≈ 2.86°
            extend_step: 每次补发延长的时长 (秒), 默认 2.0
            max_total_time: 总等待超时 (秒), 默认 60.0
        """
        import numpy as np
        import time as _time

        target = np.asarray(joints, dtype=float)
        deadline = _time.monotonic() + max_total_time
        total_sent = 0.0

        while _time.monotonic() < deadline:
            # 下发 MoveJ
            remaining = deadline - _time.monotonic()
            if total_sent == 0.0:
                # 第一次：使用原始 duration
                cur_duration = duration
            else:
                cur_duration = min(extend_step, remaining)

            if cur_duration <= 0:
                break

            self.move_joints_timed(joints, cur_duration)
            total_sent += cur_duration

            # 等待本次命令完成
            self.wait_for_hold_expire(timeout=cur_duration + 1.0)

            # 检查是否到达
            current = self.get_current_dual_arm_q()
            if len(current) < len(target):
                current = np.pad(current, (0, len(target) - len(current)))
            elif len(current) > len(target):
                current = current[:len(target)]
            error = np.max(np.abs(current - target))
            if error <= tolerance:
                return  # 到达，返回

        # 超时，打印警告
        import logging as _logging
        _logging.getLogger(__name__).warning(
            f"move_joints_timed_and_verify: 超时 {max_total_time}s, "
            f"最大关节误差 {error:.4f} rad > 容差 {tolerance}"
        )

    # ── EE ──────────────────────────────────────────────────────

    def set_ee_gripper(self, arm_idx: int, value: float) -> None:
        """设置吸盘/夹爪: 0=右, 1=左, value=0.0/1.0。默认无操作。"""
        pass

    # ── 模式感知关节采样 ────────────────────────────────────

    def get_current_joint_state(self,
                                mode: "ControlMode") -> np.ndarray:
        """按控制模式采样当前关节角 (rad)。

        ARMS_ONLY  → 14 (双臂)
        ARMS_HEAD → 16 (双臂 + 头部)
        默认基于 get_current_dual_arm_q() + get_head_q() 拼接。
        """
        from teleop.robot_control._base.control_mode import ControlMode
        arm_q = self.get_current_dual_arm_q()
        if mode == ControlMode.ARMS_ONLY:
            return np.asarray(arm_q, dtype=float)[:14]
        if mode == ControlMode.ARMS_HEAD:
            head_q = self.get_head_q()
            return np.concatenate([
                np.asarray(arm_q, dtype=float)[:14],
                np.asarray(head_q, dtype=float)[:2],
            ])
        # 预留
        if mode == ControlMode.ARMS_HEAD_TORSO:
            return np.concatenate([
                np.asarray(arm_q, dtype=float)[:14],
                np.asarray(self.get_head_q(), dtype=float)[:2],
                np.zeros(2),
            ])
        return np.asarray(arm_q, dtype=float)[:14]

    # ── 速度 ────────────────────────────────────────────────────

    def speed_gradual_max(self, t: float = 5.0) -> None:
        """5 秒内逐渐提升速度到最大值。"""
        pass

    def speed_instant_max(self) -> None:
        """立即将速度设为最大值。"""
        pass

    # ── 状态（TOPSTAR_H1 特有，其他返回 None）───────────────────

    def get_last_published_dual_arm_q(self) -> Optional[np.ndarray]:
        """获取上次下发的双臂关节角（用于可视化）。"""
        return None

    # ── 生命周期 ────────────────────────────────────────────────

    def connect(self) -> None:
        """连接硬件/DDS/ROS。子类可选覆写。"""
        pass

    def disconnect(self) -> None:
        """断开连接。子类可选覆写。"""
        pass

    @property
    def connected(self) -> bool:
        return True
