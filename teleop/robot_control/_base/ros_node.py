"""Base ROS2 节点 — 提供 LowCmd CRC 公共方法。"""

import struct
import threading

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from topstar_hg.msg import LowCmd, LowState


def _crc32_core(data: bytes) -> int:
    words = struct.unpack(f'{len(data) // 4}I', data)
    crc = 0xFFFFFFFF
    poly = 0x04C11DB7
    for word in words:
        xbit = 1 << 31
        for _ in range(32):
            if crc & 0x80000000:
                crc = ((crc << 1) & 0xFFFFFFFF) ^ poly
            else:
                crc = (crc << 1) & 0xFFFFFFFF
            if word & xbit:
                crc ^= poly
            xbit >>= 1
    return crc


class DataBuffer:
    """线程安全的最新消息缓冲区。"""

    def __init__(self):
        self._data = None
        self._timestamp_ns = 0
        self._lock = threading.Lock()

    def set(self, data, timestamp_ns=0):
        with self._lock:
            self._data = data
            self._timestamp_ns = int(timestamp_ns or 0)

    def get(self):
        with self._lock:
            return self._data

    def get_with_timestamp(self):
        with self._lock:
            return self._data, self._timestamp_ns


class BaseRosNode(Node):
    """所有机器人 ROS2 节点的基类。

    提供:
    - cmd_pub / state_sub / state_buffer
    - compute_lowcmd_crc() — 与 topstar_sdk2 兼容的 LowCmd CRC 计算
    - state_cb() 占位，子类覆写
    """

    _SENSOR_QOS = QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
    )

    def __init__(self, node_name: str, cmd_topic: str, state_topic: str):
        super().__init__(node_name)
        self.cmd_pub = self.create_publisher(LowCmd, cmd_topic, 10)
        self.state_sub = self.create_subscription(
            LowState, state_topic, self.state_cb, self._SENSOR_QOS,
        )
        self.state_buffer = None  # 子类在 __init__ 中赋值

    def state_cb(self, msg: LowState):
        """LowState 回调 — 子类覆写以添加自定义处理。"""
        raise NotImplementedError

    @staticmethod
    def compute_lowcmd_crc(msg: LowCmd) -> None:
        """计算 LowCmd CRC（与 topstar_sdk2 兼容），结果写入 msg.crc。"""
        buf = bytearray()
        buf += struct.pack('BB2x', msg.mode_pr, msg.mode_machine)
        for m in msg.motor_cmd:
            buf += struct.pack('=B3xfffffI',
                               m.mode, m.q, m.dq, m.tau, m.kp, m.kd, m.reserve)
        buf += struct.pack('4I', *list(msg.reserve))
        msg.crc = _crc32_core(bytes(buf))
