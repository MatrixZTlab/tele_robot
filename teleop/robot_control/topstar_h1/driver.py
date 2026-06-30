"""TOPSTAR_H1 遥操作驱动 — 封装 XR 预处理、IK、命令下发。"""
import logging
import numpy as np

from teleop.robot_control._base.robot_driver import RobotDriver, IKResult
from teleop.robot_control._base.control_mode import ControlMode
from teleop.robot_control.topstar_h1.config import H1RobotConfig
from teleop.robot_control.topstar_h1.arm_ik import H1ArmIK
from teleop.robot_control.topstar_h1.xr_transformer import H1XRTransformer
# H1ArmController 在 _build_components() 中惰性加载（需要 rclpy）
from teleop.robot_control.ee.suction_cup import SuctionCupHandler

logger = logging.getLogger(__name__)


class H1RobotDriver(RobotDriver):
    """TOPSTAR_H1 遥操作驱动。"""

    def __init__(self, control_mode=ControlMode.ARMS_HEAD,
                 frequency=50.0, simulation_mode=True, arm_scale=1.0,
                 ee_type=None, verbose=False):
        # ⚠️ 必须在 super().__init__() 前赋值（父类构造会调用 _build_components）
        self.ee_type = ee_type
        self.arm_scale = arm_scale
        config = H1RobotConfig()
        self.max_reach = config.arm_max_reach
        self.l_shoulder = None
        self.r_shoulder = None
        self._prev_ee_state: dict = {}

        super().__init__(config, control_mode, frequency, simulation_mode,
                         arm_scale, verbose)

    def _build_components(self):
        """创建 IK、控制器、XRTransformer。"""
        logger.info(f"Building IK solver ({self.config.model_name})...")
        self.ik = H1ArmIK(
            self.config, self.control_mode,
            visualization="ros2", verbose=self.verbose,
        )
        logger.info(f"IK ready (control DOF={self.control_mode.ik_dof})")

        logger.info("Building controller...")
        from teleop.robot_control.topstar_h1.arm_controller import H1ArmController
        self.controller = H1ArmController(
            self.config, self.control_mode,
            frequency=self.frequency, simulation_mode=self.simulation_mode,
        )
        logger.info(f"Controller ready ({type(self.controller).__name__})")

        logger.info("Building XR transformer...")
        # XR 变换器（替代 _preprocess_arm_poses / _preprocess_head）
        self.xr_transformer = H1XRTransformer(
            self.ik, self.arm_scale, self.max_reach,
        )
        self.l_shoulder = self.xr_transformer.l_shoulder
        self.r_shoulder = self.xr_transformer.r_shoulder
        logger.info("XR transformer ready")

        # EE handler
        if self.ee_type == "suction_cup":
            handler = SuctionCupHandler(arm_ctrl=self.controller)
            self.handler_registry.register(handler)
            logger.info(f"EE handler registered: {self.ee_type}")

    # ── 命令分发 ────────────────────────────────────────────

    def _dispatch_commands(self, ik_result: IKResult):
        self.controller.servo_dual_arm(ik_result.arm_q, ik_result.arm_tau)
        if (self.control_mode.solve_head
                and ik_result.head_q is not None
                and ik_result.head_q.size >= 2):
            self.controller.ctrl_head(ik_result.head_q[:2])

    def _on_step_done(self, ik_result, tele_data):
        """每帧执行 EE handler 事件分发（边缘检测 + 回调）。"""
        if self.handler_registry.count > 0:
            swap = getattr(tele_data, 'teleop_mode', 'relative_head') == 'relative_pose'
            self._prev_ee_state = self.handler_registry.dispatch_trigger_squeeze(
                tele_data, self._prev_ee_state, swap_sides=swap,
            )

    # ── 回放 ────────────────────────────────────────────────

    def replay_point(self, point, control_mode):
        """H1 回放：MoveJ→ArmRequest, ServoJ→LowCmd servo。"""
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

    # ── 录制 ────────────────────────────────────────────────

    def _collect_recording_state(self, ik_result, current_q):
        return {
            "left_arm_state":  current_q[:7].tolist(),
            "right_arm_state": current_q[-7:].tolist(),
            "left_arm_action": ik_result.arm_q[:7].tolist(),
            "right_arm_action": ik_result.arm_q[-7:].tolist(),
            "body_state": current_q.tolist(),
            "body_action": current_q.tolist(),
            **self.handler_registry.collect_recording_state(),
        }
