"""TOPSTAR_H1 机器人配置。"""
from teleop.robot_control._base.robot_config import RobotConfig


class H1RobotConfig(RobotConfig):
    """TOPSTAR_H1 静态元数据。"""

    def __init__(self, **overrides):
        defaults = dict(
            model_name="TOPSTAR_H1",
            urdf_path=(
                "/media/ai/d9787eb9-5947-4134-be08-d9b5ed71bdde"
                "/tele_robot/tele_robot_sdk/tele_robot/"
                "assets/topstar_h1/_tmp_h1_mujoco.urdf"
            ),
            model_dir=(
                "/media/ai/d9787eb9-5947-4134-be08-d9b5ed71bdde"
                "/tele_robot/tele_robot_sdk/tele_robot/"
                "assets/topstar_h1"
            ),
            locked_joint_names=[
                "Robot_Body_Movement_Joint",
                "Robot_Body_Rotation_Joint",
                "Wheel_Rotation_1_1_Joint",
                "Wheel_Rotation_1_2_Joint",
                "Wheel_Rotation_2_1_Joint",
                "Wheel_Rotation_2_2_Joint",
                "Wheel_Rotation_3_1_Joint",
                "Wheel_Rotation_3_2_Joint",
                "Wheel_Rotation_4_1_Joint",
                "Wheel_Rotation_4_2_Joint",
            ],
            left_arm_joint_names=[
                "Robot_Left_Hand_base_Joint",
                "Robot_Left_Hand_1_Joint",
                "Robot_Left_Hand_2_Joint",
                "Robot_Left_Hand_3_Joint",
                "Robot_Left_Hand_4_Joint",
                "Robot_Left_Hand_5_Joint",
                "Robot_Left_Hand_6_Joint",
            ],
            right_arm_joint_names=[
                "Robot_Right_Hand_base_Joint",
                "Robot_Right_Hand_1_Joint",
                "Robot_Right_Hand_2_Joint",
                "Robot_Right_Hand_3_Joint",
                "Robot_Right_Hand_4_Joint",
                "Robot_Right_Hand_5_Joint",
                "Robot_Right_Hand_6_Joint",
            ],
            left_ee_frame_name="io_teleop_left_ee_link",
            right_ee_frame_name="io_teleop_right_ee_link",
            head_joint_names=[
                "Robot_Head_Rotation_Joint",
                "Robot_Head_Tonod_Joint",
            ],
            head_frame_name="Robot_Head_Tonod_Link",
            head_control_method="ik",
            torso_joint_names=[
                "Robot_Body_Movement_Joint",
                "Robot_Body_Rotation_Joint",
            ],
            limit_mode="modified",
            cache_filename="topstar_h1_model_cache.pkl",
            arm_max_reach=0.69,
            arm_scale_enabled=True,
            has_visualization=True,
            wrist_joint_order="roll_pitch_yaw",
            supported_ees=["suction_cup"],
        )
        defaults.update(overrides)
        super().__init__(**defaults)
