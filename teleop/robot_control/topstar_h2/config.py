"""TOPSTAR_H2 机器人配置。"""
from pathlib import Path

from teleop.robot_control._base.robot_config import RobotConfig

_REPO_ROOT = Path(__file__).resolve().parents[3]
_MODEL_DIR = _REPO_ROOT / "assets" / "TH010_URDF_V2.1" / "urdf"


class H2RobotConfig(RobotConfig):
    """TOPSTAR_H2 静态元数据（基于 TH010 URDF）。"""

    def __init__(self, **overrides):
        defaults = dict(
            model_name="TOPSTAR_H2",
            urdf_path=str(_MODEL_DIR / "TH010_URDF_V2.0.urdf"),
            model_dir=str(_MODEL_DIR),
            locked_joint_names=[
                # 双腿 12 关节 + 腰部 1 关节
                "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
                "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
                "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
                "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
                "waist_yaw_joint",
            ],
            left_arm_joint_names=[
                "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
                "left_shoulder_yaw_joint", "left_elbow_joint",
                "left_wrist_yaw_joint", "left_wrist_pitch_joint",
                "left_wrist_roll_joint",
            ],
            right_arm_joint_names=[
                "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
                "right_shoulder_yaw_joint", "right_elbow_joint",
                "right_wrist_yaw_joint", "right_wrist_pitch_joint",
                "right_wrist_roll_joint",
            ],
            left_ee_frame_name="L_ee",
            right_ee_frame_name="R_ee",
            head_joint_names=[
                "head_yaw_joint",
                "head_pitch_joint",
            ],
            head_frame_name="head_pitch_link",
            head_control_method="ik",
            torso_joint_names=None,
            limit_mode="modified",
            cache_filename="topstar_h2_model_cache.pkl",
            arm_max_reach=0.69,
            arm_scale_enabled=True,
            has_visualization=True,
            wrist_joint_order="yaw_pitch_roll",
            supported_ees=[],
        )
        defaults.update(overrides)
        super().__init__(**defaults)
