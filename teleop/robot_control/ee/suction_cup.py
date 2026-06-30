from __future__ import annotations

from teleop.robot_control.handler_base import BaseHandler

import logging_mp
logger_mp = logging_mp.getLogger(__name__)


class SuctionCupHandler(BaseHandler):
    """吸盘式末端治具 handler。

    Button-to-action mapping:
      - Left  trigger → toggle left  suction ON/OFF
      - Right trigger → toggle right suction ON/OFF
      - squeeze keys are unused.

    状态由 handler 自持，不再依赖 arm_ctrl._ee_gripper_state。
    """

    def __init__(self, arm_ctrl=None):
        super().__init__()
        self.arm_ctrl = arm_ctrl
        self._suction_state: list[bool] = [False, False]  # [right, left]

    def _toggle(self, arm_idx: int, side_label: str):
        self._suction_state[arm_idx] = not self._suction_state[arm_idx]
        new_val = 1.0 if self._suction_state[arm_idx] else 0.0
        if self.arm_ctrl is not None:
            self.arm_ctrl.set_ee_gripper(arm_idx=arm_idx, value=new_val)
        logger_mp.info(
            f"[SuctionCup] {side_label} suction → "
            f"{'ON' if self._suction_state[arm_idx] else 'OFF'}"
        )

    # ── 触发 ────────────────────────────────────────────────

    def on_left_trigger_press(self):
        self._toggle(arm_idx=1, side_label="Left")

    def on_right_trigger_press(self):
        self._toggle(arm_idx=0, side_label="Right")

    # squeeze keys remain unused
    # release events are ignored (toggle happens on press only)

    # ── 录制快照 ────────────────────────────────────────────

    def get_state_for_recording(self) -> dict:
        return {
            "ee_action": [float(self._suction_state[0]),
                          float(self._suction_state[1])],
        }
