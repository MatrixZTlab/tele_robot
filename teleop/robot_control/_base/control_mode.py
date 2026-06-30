from __future__ import annotations

from enum import Enum


class ControlMode(str, Enum):
    """控制模式 — 决定 IK 优化变量维度和命令下发范围。"""

    ARMS_ONLY = "arms_only"
    """仅双臂 14DOF（当前 solve_head=False）。"""
    ARMS_HEAD = "arms_head"
    """双臂 + 头部 16DOF（当前 solve_head=True）。"""
    ARMS_HEAD_TORSO = "arms_head_torso"
    """双臂 + 头部 + 躯干 18DOF（预留）。"""
    FULL_BODY = "full_body"
    """全身控制（含底盘轮子）35DOF（预留）。"""

    # ── 属性 ────────────────────────────────────────────────────

    @property
    def ik_dof(self) -> int:
        """IK 优化变量维度。"""
        if self == ControlMode.ARMS_ONLY:
            return 14
        elif self == ControlMode.ARMS_HEAD:
            return 16
        elif self == ControlMode.ARMS_HEAD_TORSO:
            return 18
        elif self == ControlMode.FULL_BODY:
            return 35
        raise ValueError(f"Unknown control mode: {self!r}")

    @property
    def solve_head(self) -> bool:
        """头部是否作为 IK 优化变量。"""
        return self in (
            ControlMode.ARMS_HEAD,
            ControlMode.ARMS_HEAD_TORSO,
            ControlMode.FULL_BODY,
        )

    @property
    def solve_torso(self) -> bool:
        """躯干是否作为 IK 优化变量。"""
        return self in (ControlMode.ARMS_HEAD_TORSO, ControlMode.FULL_BODY)

    @property
    def solve_wheels(self) -> bool:
        """底盘轮子是否作为优化变量。"""
        return self == ControlMode.FULL_BODY

    # ── 工厂 ────────────────────────────────────────────────────

    @staticmethod
    def from_str(value: str) -> ControlMode:
        """从字符串安全解析（不区分大小写、下划线/连字符兼容）。"""
        normalized = value.strip().lower().replace("-", "_")
        for mode in ControlMode:
            if mode.value == normalized:
                return mode
        raise ValueError(
            f"Unsupported control mode: {value!r}. "
            f"Options: {[m.value for m in ControlMode]}"
        )
