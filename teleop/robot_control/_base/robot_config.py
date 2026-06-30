from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RobotConfig:
    """机器人静态元数据 — 每个机器人子类填充自己的值。"""

    # ── 标识 ──
    model_name: str  # e.g. "TOPSTAR_H1", "TOPSTAR_H2"

    # ── URDF ──
    urdf_path: str
    model_dir: str
    locked_joint_names: list[str] = field(default_factory=list)
    left_arm_joint_names: list[str] = field(default_factory=lambda: [f"L_{i}" for i in range(7)])
    right_arm_joint_names: list[str] = field(default_factory=lambda: [f"R_{i}" for i in range(7)])

    # ── EE frame ──
    left_ee_frame_name: str = "L_ee"
    right_ee_frame_name: str = "R_ee"

    # ── 头部 ──
    head_joint_names: Optional[list[str]] = None
    head_frame_name: Optional[str] = None
    head_control_method: str = "none"
    """头部控制方式: "ik" | "euler" | "none" """

    # ── 躯干 ──
    torso_joint_names: Optional[list[str]] = None

    # ── 限位 ──
    limit_mode: str = "urdf"
    """限位来源: "urdf" | "modified" """

    # ── 缓存 ──
    cache_filename: str = "model_cache.pkl"

    # ── 臂展预处理 ──
    arm_max_reach: Optional[float] = None
    """单臂最大伸展半径（米），None=不限幅。"""
    arm_scale_enabled: bool = False
    """是否启用基于肩部的臂展缩放。"""

    # ── 可视化 ──
    has_visualization: bool = False

    # ── WS 关节顺序（仅 TOPSTAR_H2 需要） ──
    wrist_joint_order: str = "roll_pitch_yaw"
    """腕关节顺序: "roll_pitch_yaw" | "yaw_pitch_roll" """

    # ── EE 能力 ──
    supported_ees: list[str] = field(default_factory=list)
    """支持的末端执行器类型列表，如 ["suction_cup"]"""

    def supports_ee(self, ee_type: Optional[str]) -> bool:
        """检查是否支持指定的 EE 类型。None 或空字符串表示无 EE。"""
        if not ee_type:
            return True
        return ee_type in self.supported_ees
