"""TOPSTAR_H2 遥操作驱动 — 封装 XR 变换、IK、命令下发。"""
import logging
import numpy as np

from teleop.robot_control._base.robot_driver import RobotDriver, IKResult
from teleop.robot_control._base.control_mode import ControlMode
from teleop.robot_control.topstar_h2.config import H2RobotConfig
from teleop.robot_control.topstar_h2.arm_ik import H2ArmIK
from teleop.robot_control.topstar_h2.xr_transformer import H2XRTransformer
# H2ArmController 在 _build_components() 中惰性加载（需要 rclpy）

logger = logging.getLogger(__name__)


class H2RobotDriver(RobotDriver):
    """TOPSTAR_H2 遥操作驱动。"""

    def __init__(self, control_mode=ControlMode.ARMS_HEAD,
                 frequency=50.0, simulation_mode=True, arm_scale=1.0,
                 ee_type=None, verbose=False):
        self.ee_type = ee_type
        self.arm_scale = arm_scale
        config = H2RobotConfig()
        self.max_reach = config.arm_max_reach
        self._prev_ee_state: dict = {}

        super().__init__(config, control_mode, frequency, simulation_mode,
                         arm_scale, verbose)

    def _build_components(self):
        """创建 IK、控制器、XRTransformer。"""
        logger.info(f"Building IK solver ({self.config.model_name})...")
        self.ik = H2ArmIK(
            self.config, self.control_mode,
            visualization="off", verbose=self.verbose,
        )
        logger.info(f"IK ready (control DOF={self.control_mode.ik_dof})")

        logger.info("Building controller...")
        from teleop.robot_control.topstar_h2.arm_controller import H2ArmController
        self.controller = H2ArmController(
            self.config, self.control_mode,
            frequency=self.frequency, simulation_mode=self.simulation_mode,
        )
        logger.info(f"Controller ready ({type(self.controller).__name__})")

        logger.info("Building XR transformer...")
        # XR 变换器（替代 _preprocess_arm_poses / _preprocess_head）
        self.xr_transformer = H2XRTransformer(
            self.ik, self.arm_scale, self.max_reach,
        )
        logger.info("XR transformer ready")

    # ── 命令分发 ────────────────────────────────────────────

    def _dispatch_commands(self, ik_result: IKResult):
        self.controller.servo_dual_arm(ik_result.arm_q, ik_result.arm_tau)
        if (self.control_mode.solve_head
                and ik_result.head_q is not None
                and ik_result.head_q.size >= 2):
            self.controller.ctrl_head(ik_result.head_q[:2])

    # ── 回放 ────────────────────────────────────────────────

    def replay_point(self, point, control_mode):
        """H2 回放：全部走 LowCmd 位置模式（MoveJ→插值, ServoJ→servo）。"""
        import numpy as np
        point_type = point.get("type")
        joint_pose_deg = point.get("joint_pose")
        if joint_pose_deg is None:
            return

        joint_pose_rad = np.deg2rad(
            np.asarray(joint_pose_deg, dtype=float)
        ).tolist()

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
        if control_mode.solve_head:
            if len(joint_pose_rad) >= 16:
                self.controller.ctrl_head(joint_pose_rad[14:16])
            elif point.get("head_pose") is not None:
                head_rad = np.deg2rad(
                    np.asarray(point["head_pose"], dtype=float)
                )[:2].tolist()
                self.controller.ctrl_head(head_rad)

    # ── 录制（无 EE handler） ───────────────────────────────

    def _collect_recording_state(self, ik_result, current_q):
        return {
            "left_arm_state":  current_q[:7].tolist(),
            "right_arm_state": current_q[-7:].tolist(),
            "left_arm_action": ik_result.arm_q[:7].tolist(),
            "right_arm_action": ik_result.arm_q[-7:].tolist(),
            "body_state": current_q.tolist(),
            "body_action": current_q.tolist(),
        }
