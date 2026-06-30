"""工厂函数 — 机器人无关的 create_robot_driver()。"""
from teleop.robot_control._base.control_mode import ControlMode
from teleop.robot_control._base.robot_driver import RobotDriver


ROBOT_REGISTRY: dict[str, type[RobotDriver]] = {}
"""已注册的机器人驱动类。键为 model_name。"""


def register_robot(model_name: str, driver_cls: type[RobotDriver]):
    """注册一个机器人驱动类。"""
    ROBOT_REGISTRY[model_name.upper()] = driver_cls


def create_robot_driver(model_name: str, **kwargs) -> RobotDriver:
    """创建机器人驱动实例。

    Args:
        model_name: "TOPSTAR_H1" | "TOPSTAR_H2"
        **kwargs: 传递给驱动构造函数的参数。

    Returns:
        RobotDriver 实例。

    Raises:
        ValueError: model_name 未注册。
    """
    key = model_name.upper()
    if key not in ROBOT_REGISTRY:
        # 首次访问时尝试惰性注册
        _lazy_register(key)
    if key not in ROBOT_REGISTRY:
        raise ValueError(
            f"Unknown robot: {model_name!r}. "
            f"Registered: {list(ROBOT_REGISTRY)}"
        )
    return ROBOT_REGISTRY[key](**kwargs)


# ── 惰性注册 ────────────────────────────────────────────────
_LAZY_MAP = {
    "TOPSTAR_H1": "teleop.robot_control.topstar_h1.driver:H1RobotDriver",
    "TOPSTAR_H2": "teleop.robot_control.topstar_h2.driver:H2RobotDriver",
}


def _lazy_register(key: str):
    if key not in _LAZY_MAP:
        return
    module_path, class_name = _LAZY_MAP[key].split(":")
    try:
        import importlib
        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name)
        register_robot(key, cls)
    except ImportError:
        pass  # 可选依赖（如 rclpy），不强制

__all__ = [
    "register_robot",
    "create_robot_driver",
    "ROBOT_REGISTRY",
]
