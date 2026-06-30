# teleop/robot_control/__init__.py
# 顶层导出 — 仅导入轻量级模块，CasADi/Pinocchio 惰性加载

from teleop.robot_control._base.control_mode import ControlMode
from teleop.robot_control._base.robot_config import RobotConfig
from teleop.robot_control._base.arm_controller import BaseArmController
from teleop.robot_control._base.robot_driver import RobotDriver
from teleop.robot_control.handler_base import BaseHandler
from teleop.robot_control.handler_registry import HandlerRegistry

# BaseArmIK 和 IKResult 可惰性导入，由各子类自动触发
# from teleop.robot_control._base.arm_ik import BaseArmIK, IKResult

# 工厂函数（在 factory.py 中按需导入）
# from teleop.robot_control.factory import create_robot_driver

__all__ = [
    "ControlMode",
    "RobotConfig",
    "BaseArmController",
    "RobotDriver",
    "BaseHandler",
    "HandlerRegistry",
]
