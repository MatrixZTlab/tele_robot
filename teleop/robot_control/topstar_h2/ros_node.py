"""TOPSTAR_H2 ROS2 节点。"""
import logging
import threading

import rclpy
from topstar_hg.msg import LowCmd, LowState

from teleop.robot_control._base.ros_node import BaseRosNode, DataBuffer

logger = logging.getLogger(__name__)


class H2RosNode(BaseRosNode):
    """TOPSTAR_H2 ROS2 节点 — 话题无 /h1/ 前缀，29 电机。"""

    _NUM_MOTORS = 29

    def __init__(self):
        super().__init__(node_name='h2_arm_controller',
                         cmd_topic='/lowcmd',
                         state_topic='/lowstate')
        self.state_buffer = DataBuffer()

        # ── 电机故障状态 ──
        self.fault_buffer = DataBuffer()

        # ── Sport API (FSM 切换) ──
        from topstar_api.msg import Request, Response
        self._sport_req_pub = self.create_publisher(
            Request, '/api/sport/request', 10,
        )
        self._sport_resp_buffer = DataBuffer()
        self._sport_resp_event = threading.Event()
        self._sport_resp_sub = self.create_subscription(
            Response, '/api/sport/response',
            self._sport_resp_cb,
            self._SENSOR_QOS,
        )

    def _sport_resp_cb(self, msg):
        self._sport_resp_buffer.set(msg)
        self._sport_resp_event.set()
        # 监控 Sport API 响应中的错误码
        if msg.header.status.code != 0:
            logger.warning(
                f"[SportAPI Error] id={msg.header.identity.id}, "
                f"api_id={msg.header.identity.api_id}, "
                f"code={msg.header.status.code}, "
                f"data={msg.data}"
            )

    def state_cb(self, msg: LowState):
        self.state_buffer.set(msg)

        # ── 电机故障检测 ────────────────────────────────────────────
        # 参考: h2_joint_oscillation_example.cpp LowStateHandler
        #
        # Primary:   per-motor motorstate 字段 (non-zero = EtherCAT fault code)
        # Secondary: reserve[0] system flag packed by topstar_bridge
        #   bits 0-7  = motor_fault_active
        #   bits 8-15 = faulted_motor_id
        sys_fault = (msg.reserve[0] & 0xFF) != 0
        sys_fault_id = (msg.reserve[0] >> 8) & 0xFF

        per_motor_fault_id = -1
        for i in range(self._NUM_MOTORS):
            if msg.motor_state[i].motorstate != 0:
                per_motor_fault_id = i
                break

        fault_now = sys_fault or (per_motor_fault_id >= 0)

        prev_fault = self.fault_buffer.get()
        prev_active = prev_fault.get("active", False) if prev_fault else False

        if fault_now != prev_active:
            if fault_now:
                fid = sys_fault_id if sys_fault else per_motor_fault_id
                ms_val = msg.motor_state[fid].motorstate if (0 <= fid < self._NUM_MOTORS) else 0
                logger.error(
                    f"[H2 FAULT] Motor {fid} fault (motorstate=0x{ms_val:08X}, "
                    f"reserve[0]=0x{msg.reserve[0]:08X}) — "
                    f"stopping LowCmd publish"
                )
                self.fault_buffer.set({
                    "active": True,
                    "motor_id": fid,
                    "motorstate": ms_val,
                    "reserve0": msg.reserve[0],
                })
            else:
                logger.warning("[H2 FAULT] Motor fault cleared — resuming LowCmd publish")
                self.fault_buffer.set({
                    "active": False,
                    "motor_id": -1,
                    "motorstate": 0,
                    "reserve0": 0,
                })

        # ── FSM 状态变化监控 ──
        if not hasattr(self, '_prev_fsm'):
            self._prev_fsm = msg.mode_machine
        if msg.mode_machine != self._prev_fsm:
            logger.warning(f"[H2 FSM] {self._prev_fsm} → {msg.mode_machine}")
            self._prev_fsm = msg.mode_machine
