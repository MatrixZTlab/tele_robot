from __future__ import annotations

from typing import Any

from teleop.robot_control.handler_base import BaseHandler


class HandlerRegistry:
    """管理多个 handler，统一调度 XR 输入和录制快照。"""

    def __init__(self):
        self._handlers: list[BaseHandler] = []

    # ── 注册 ────────────────────────────────────────────────────

    def register(self, handler: BaseHandler) -> None:
        """注册一个 handler。"""
        self._handlers.append(handler)

    def clear(self) -> None:
        """清空所有 handler。"""
        self._handlers.clear()

    @property
    def count(self) -> int:
        return len(self._handlers)

    # ── XR 事件分发 ─────────────────────────────────────────────

    def dispatch_trigger_squeeze(self, tele_data: Any,
                                 prev_state: dict[str, bool] | None = None,
                                 swap_sides: bool = False,
                                 ) -> dict[str, bool]:
        """将 XR trigger/squeeze 事件分发给所有启用的 handler。

        参数:
            swap_sides: 是否交换左右 trigger 映射。
                        relative_pose 模式下应为 True（左手→右臂 DO）。

        返回更新后的 prev_state 字典，供下一帧使用。
        """
        if prev_state is None:
            prev_state = {}

        for h in self._handlers:
            if not h.enabled:
                continue

            # ── left trigger ──
            left_trig = getattr(tele_data, 'left_ctrl_triggerValue', 10.0) < 5.0
            key_lt = f"{id(h)}_left_trig"
            prev_lt = prev_state.get(key_lt, False)
            if left_trig and not prev_lt:
                h.on_right_trigger_press() if swap_sides else h.on_left_trigger_press()
            elif not left_trig and prev_lt:
                h.on_right_trigger_release() if swap_sides else h.on_left_trigger_release()
            prev_state[key_lt] = left_trig

            # ── left squeeze ──
            left_sqz = getattr(tele_data, 'left_ctrl_squeezeValue', 0.0) > 0.5
            key_ls = f"{id(h)}_left_sqz"
            prev_ls = prev_state.get(key_ls, False)
            if left_sqz and not prev_ls:
                h.on_right_squeeze_press() if swap_sides else h.on_left_squeeze_press()
            elif not left_sqz and prev_ls:
                h.on_right_squeeze_release() if swap_sides else h.on_left_squeeze_release()
            prev_state[key_ls] = left_sqz

            # ── right trigger ──
            right_trig = getattr(tele_data, 'right_ctrl_triggerValue', 10.0) < 5.0
            key_rt = f"{id(h)}_right_trig"
            prev_rt = prev_state.get(key_rt, False)
            if right_trig and not prev_rt:
                h.on_left_trigger_press() if swap_sides else h.on_right_trigger_press()
            elif not right_trig and prev_rt:
                h.on_left_trigger_release() if swap_sides else h.on_right_trigger_release()
            prev_state[key_rt] = right_trig

            # ── right squeeze ──
            right_sqz = getattr(tele_data, 'right_ctrl_squeezeValue', 0.0) > 0.5
            key_rs = f"{id(h)}_right_sqz"
            prev_rs = prev_state.get(key_rs, False)
            if right_sqz and not prev_rs:
                h.on_left_squeeze_press() if swap_sides else h.on_right_squeeze_press()
            elif not right_sqz and prev_rs:
                h.on_left_squeeze_release() if swap_sides else h.on_right_squeeze_release()
            prev_state[key_rs] = right_sqz

        return prev_state

    # ── 录制快照聚合 ────────────────────────────────────────────

    def collect_recording_state(self) -> dict[str, Any]:
        """聚合所有 handler 的录制快照。"""
        result: dict[str, Any] = {}
        for h in self._handlers:
            result.update(h.get_state_for_recording())
        return result

    # ── 生命周期 ────────────────────────────────────────────────

    def on_teleop_start(self) -> None:
        for h in self._handlers:
            h.on_teleop_start()

    def on_teleop_stop(self) -> None:
        for h in self._handlers:
            h.on_teleop_stop()
