"""TOPSTAR_H1 ROS2 节点 — 从旧 topstar_h1_sim_arm_controller.py 提取。"""
import json
import threading

import rclpy
from geometry_msgs.msg import Twist
from topstar_hg.msg import LowCmd, LowState, MotorCmd, GripperCmd
from topstar_api.msg import Request as ArmRequest

from teleop.robot_control._base.ros_node import BaseRosNode, DataBuffer


# ── ROS2 节点 ───────────────────────────────────────────────

class H1RosNode(BaseRosNode):
    """TOPSTAR_H1 的 ROS2 节点，管理所有话题通信。"""

    def __init__(self):
        super().__init__(node_name='h1_arm_controller',
                         cmd_topic='/lowcmd',
                         state_topic='/lowstate')
        self.base_cmd_pub = self.create_publisher(Twist, '/base_cmd', 10)
        self.movej_pub = self.create_publisher(ArmRequest, "/api/arm/request", 10)
        self._movej_req_id = 0
        self._gripper_right_pub = self.create_publisher(GripperCmd, '/hand/right/cmd', 10)
        self._gripper_left_pub = self.create_publisher(GripperCmd, '/hand/left/cmd', 10)

        self.state_buffer = DataBuffer()

    def state_cb(self, msg: LowState):
        self.state_buffer.set(msg)

    def publish_base_cmd(self, vx: float, vy: float, vyaw: float):
        msg = Twist()
        msg.linear.x = float(vx)
        msg.linear.y = float(vy)
        msg.angular.z = float(vyaw)
        self.base_cmd_pub.publish(msg)

    def publish_movej_request(self, joints: list[float], duration: float):
        self._movej_req_id += 1
        msg = ArmRequest()
        msg.header.identity.api_id = 1001  # ARM_API_ID_MOVE_JOINTS_TIMED
        msg.header.identity.id = self._movej_req_id
        msg.parameter = json.dumps({"joints": joints, "duration": duration})
        self.movej_pub.publish(msg)

    def publish_gripper_cmd(self, arm_idx: int, position: float, mode: int = 1):
        """发布 GripperCmd 到对应手臂的 gripper 话题。

        Args:
            arm_idx: 0=右臂 (right), 1=左臂 (left)
            position: 0.0=关闭, 1.0=打开
            mode: GripperCmd 模式 (默认 1, 与 jog 面板一致)
        """
        pub = self._gripper_right_pub if arm_idx == 0 else self._gripper_left_pub
        msg = GripperCmd()
        msg.position = float(position)
        msg.mode = mode
        pub.publish(msg)
