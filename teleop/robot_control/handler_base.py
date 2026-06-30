from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional


class BaseHandler(ABC):
    """所有 handler 的基类 — 处理 XR 输入、维护设备状态、提供录制快照。"""

    def __init__(self):
        self._enabled = True

    @property
    def enabled(self) -> bool:
        return self._enabled

    def enable(self) -> None:
        self._enabled = True

    def disable(self) -> None:
        self._enabled = False

    # ── XR 输入事件（子类按需覆写） ──────────────────────────

    def on_left_trigger_press(self) -> None:
        pass

    def on_left_trigger_release(self) -> None:
        pass

    def on_left_squeeze_press(self) -> None:
        pass

    def on_left_squeeze_release(self) -> None:
        pass

    def on_right_trigger_press(self) -> None:
        pass

    def on_right_trigger_release(self) -> None:
        pass

    def on_right_squeeze_press(self) -> None:
        pass

    def on_right_squeeze_release(self) -> None:
        pass

    # ── 录制快照 ────────────────────────────────────────────────

    def get_state_for_recording(self) -> dict[str, Any]:
        """返回 handler 当前状态的字典，供录制器写入轨迹。默认空。"""
        return {}

    # ── 生命周期 ────────────────────────────────────────────────

    def on_teleop_start(self) -> None:
        pass

    def on_teleop_stop(self) -> None:
        pass

    def on_pre_ik(self, tele_data: Any) -> None:
        """在 IK 求解前调用，可用于处理 XR 手部数据。"""
        pass

    def on_post_command(self) -> None:
        """在命令下发后调用。"""
        pass
