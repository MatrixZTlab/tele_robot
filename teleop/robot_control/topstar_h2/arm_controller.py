"""TOPSTAR_H2 机械臂控制器 — 基于 ROS2，事件驱动发布，29 电机，异步插值 MoveJ。"""
import logging
import threading
import time
import traceback
from pathlib import Path
import numpy as np

import rclpy
from topstar_hg.msg import LowCmd, LowState, MotorCmd

from teleop.robot_control._base.arm_controller import BaseArmController
from teleop.robot_control._base.control_mode import ControlMode
from teleop.robot_control.topstar_h2.ros_node import (
    H2RosNode, DataBuffer,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(name)s %(message)s')
logger = logging.getLogger(__name__)

# ── H2 PD 增益 ─────────────────────────────────────────────
# 腿+躯干(0-12)=100, 头+臂(13-28)=50
_H2_KP = [100.0] * 13 + [50.0] * 16   # 29 个
_H2_KD = [1.0] * 29


class H2ArmController(BaseArmController):
    """TOPSTAR_H2 机械臂控制器 — 基于 ROS2，事件驱动发布。

    话题: /lowcmd, /lowstate
    电机数: 29
    槽位: left_arm 15-21, right_arm 22-28, head 13-14
    """

    def __init__(self, config, control_mode=ControlMode.ARMS_HEAD,
                 frequency=50.0, simulation_mode=True):
        super().__init__(config, control_mode, frequency, simulation_mode)
        logger.info("H2 ArmController: initializing...")

        self.dt = 1.0 / self.frequency
        self.dq_limit = 20.0  # rad/s 默认
        self.head_yaw_limit = (-1.6057, 1.6057)
        self.head_pitch_limit = (-0.2967, 0.384)
        self.head_dq_limit = 15.0
        self.head_smooth_alpha = 0.8

        # 槽位映射
        self.left_slots = list(range(15, 22))
        self.right_slots = list(range(22, 29))
        self.head_slots = [13, 14]
        self._motor_count = 29

        # 命令锁和目标
        self.publish_lock = threading.Lock()
        self.q_target = np.zeros(14)
        self.tauff_target = np.zeros(14)
        self.last_published_q = np.zeros(14)
        self.head_target = np.zeros(2)
        self.last_published_head_q = np.zeros(2)

        # ROS2
        if not rclpy.ok():
            rclpy.init(args=None)
        self._ros_node = H2RosNode()
        logger.info("H2 ArmController: ROS2 node created")

        # 发布控制
        self.publish_event = threading.Event()
        self._shutdown_event = threading.Event()
        self._min_publish_interval = 0.002  # H2: 最短 2ms
        self._last_publish_time = 0.0
        self._arm_active = False
        self._gradual_start_time = None
        self._dq_limit_override = None
        self._speed_gradual_max = False

        # ── MoveJ 插值器状态 ──
        self._movej_active = False
        self._movej_start_q = np.zeros(14)
        self._movej_target_q = np.zeros(14)
        self._movej_start_time = 0.0
        self._movej_duration = 0.0
        self._joint_hold_until = 0.0

        # ── 命令日志文件 ──
        self._command_log_file = None
        self._log_path = Path(__file__).resolve().parents[3] / 'Log' / 'h2_arm_controller_cmd.log'
        try:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            self._command_log_file = self._log_path.open('a', encoding='utf-8')
            logger.info(f"Command log: {self._log_path}")
        except Exception:
            logger.warning("Failed to open command log file")

        # ── Manual mode (FSM_MANUAL) ──
        self._manual_mode_ready = False
        self._manual_thread = threading.Thread(
            target=self._ensure_manual_mode, daemon=True,
        )
        self._manual_thread.start()

        # 线程
        logger.info("H2 ArmController: starting spin thread...")
        self.spin_thread = threading.Thread(target=rclpy.spin, args=(self._ros_node,), daemon=True)
        self.spin_thread.start()
        logger.info(f"H2 ArmController: starting publish loop ({self.frequency} Hz)...")
        self.publish_thread = threading.Thread(target=self._publish_loop, daemon=True)
        self.publish_thread.start()

    # ══════════════════════════════════════════════════════════
    #  BaseArmController 接口
    # ══════════════════════════════════════════════════════════

    def servo_dual_arm(self, q_target, tauff=None):
        """设置双臂伺服目标 (rad, sim convention)。"""
        self._arm_active = True
        values = np.asarray(q_target, dtype=float).flatten()
        tau = np.asarray(tauff, dtype=float).flatten() if tauff is not None else np.zeros(14)
        with self.publish_lock:
            self.q_target = values
            self.tauff_target = tau
        self.publish_event.set()

    def get_current_dual_arm_q(self):
        state = self._ros_node.state_buffer.get()
        if state is None:
            return np.zeros(14)
        res = np.zeros(14)
        for i, idx in enumerate(self.left_slots):
            res[i] = state.motor_state[idx].q
        for i, idx in enumerate(self.right_slots):
            res[7 + i] = state.motor_state[idx].q
        return res  # 直通，无 sign_map

    def get_current_dual_arm_dq(self):
        state = self._ros_node.state_buffer.get()
        if state is None:
            return np.zeros(14)
        res = np.zeros(14)
        for i, idx in enumerate(self.left_slots):
            res[i] = state.motor_state[idx].dq
        for i, idx in enumerate(self.right_slots):
            res[7 + i] = state.motor_state[idx].dq
        return res  # 直通

    def ctrl_head(self, head_q):
        """控制头部 (IK convention: [yaw, pitch])。直通无坐标转换。"""
        with self.publish_lock:
            h = np.asarray(head_q)
            if h.size >= 2:
                new_target = np.array([
                    float(np.clip(h[0], *self.head_yaw_limit)),
                    float(np.clip(h[1], *self.head_pitch_limit)),
                ], dtype=float)
                self.head_target = (
                    self.head_smooth_alpha * new_target
                    + (1.0 - self.head_smooth_alpha) * self.head_target
                )
        self.publish_event.set()

    def move_joints_timed(self, joints, duration):
        """H2 MoveJ：设置插值目标，立即返回，_publish_loop 异步执行。"""
        self._arm_active = True
        values = np.asarray(joints, dtype=float).flatten()[:14]
        # 使用机器人当前实际关节角作为插值起点，避免首次调用时
        # last_published_q 为全零导致瞬间回零再出发的问题
        current_actual = self.get_current_dual_arm_q()
        with self.publish_lock:
            self._movej_start_q = current_actual.copy()
            self._movej_target_q = values.copy()
            self._movej_start_time = time.monotonic()
            self._movej_duration = float(duration)
            self._movej_active = True
            # 同步 servo 目标，方便 wait_for_hold_expire 后立即进入常态
            self.q_target = values.copy()
            self.tauff_target = np.zeros(14)
            self.last_published_q = current_actual.copy()
        self._joint_hold_until = time.monotonic() + float(duration)
        self.publish_event.set()

    def wait_for_hold_expire(self, timeout=10.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < self._joint_hold_until:
            if time.monotonic() > deadline:
                break
            time.sleep(0.01)

    def go_home(self, timeout=10.0):
        """H2 通过 MoveJ 插值归零（替代瞬切 servo）。"""
        self.move_joints_timed(
            np.array([-1.6, 0, 0, 0, 0, 0, 0, -1.6, 0, 0, 0, 0, 0, 0], dtype=float),
            duration=3.0,
        )

    def get_current_joint_state(self, mode):
        """H2: ARMS_HEAD 时拼接 14 arm + 2 head = 16 DOF。"""
        arm_q = self.get_current_dual_arm_q()
        if mode == ControlMode.ARMS_HEAD:
            head_q = self.get_head_q()
            return np.concatenate([np.asarray(arm_q, dtype=float)[:14],
                                   np.asarray(head_q, dtype=float)[:2]])
        return np.asarray(arm_q, dtype=float)[:14]

    def get_head_q(self):
        state = self._ros_node.state_buffer.get()
        if state is None:
            return np.zeros(2)
        try:
            return np.array([
                state.motor_state[self.head_slots[0]].q,
                state.motor_state[self.head_slots[1]].q,
            ], dtype=float)
        except Exception:
            return np.zeros(2)

    def speed_gradual_max(self, t=5.0):
        self._speed_gradual_max = True
        self._gradual_start_time = time.perf_counter()
        self._dq_limit_override = None

    def speed_instant_max(self):
        self._speed_gradual_max = False
        self._dq_limit_override = 30.0
        self.dq_limit = 30.0

    def connect(self):
        pass

    def disconnect(self):
        self.stop()

    @property
    def connected(self) -> bool:
        return rclpy.ok()

    # ══════════════════════════════════════════════════════════
    #  Manual mode (FSM_MANUAL 切换)
    # ══════════════════════════════════════════════════════════

    def _ensure_manual_mode(self):
        """通过 rt/api/sport/request 切换 H2 到 FSM_MANUAL (fsm_id=9)。

        仿真模式下跳过（sport API 话题不存在）。
        使用 identity.id 精确匹配 response，最多重试 3 次。
        """
        # 仿真模式：跳过 manual 切换，直接可用
        if self.simulation_mode:
            self._manual_mode_ready = True
            logger.info("[H2] Simulation mode — skipping FSM_MANUAL switch")
            return

        import json as _json
        from topstar_api.msg import Request

        api_id = 7101  # ROBOT_API_ID_LOCO_SET_FSM_ID
        param = _json.dumps({"data": 9})  # FSM_MANUAL

        max_retries = 3
        timeout = 5.0

        for attempt in range(max_retries):
            req_id = int(time.monotonic_ns())
            req = Request()
            req.header.identity.id = req_id
            req.header.identity.api_id = api_id
            req.header.policy.noreply = False  # 需要响应
            req.parameter = param

            self._ros_node._sport_resp_event.clear()
            self._ros_node._sport_req_pub.publish(req)

            matched = False
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                if self._ros_node._sport_resp_event.wait(
                        timeout=deadline - time.monotonic()):
                    resp = self._ros_node._sport_resp_buffer.get()
                    if (resp is not None
                            and resp.header.identity.id == req_id):
                        matched = True
                        code = resp.header.status.code
                        if code == 0:
                            self._manual_mode_ready = True
                            logger.info(
                                "[H2] FSM_MANUAL (fsm_id=9) confirmed OK"
                            )
                            return
                        logger.warning(
                            f"[H2] Manual mode rejected "
                            f"(attempt {attempt+1}/{max_retries}, "
                            f"code={code})"
                        )
                        break
                    self._ros_node._sport_resp_event.clear()

            if not matched:
                logger.warning(
                    f"[H2] No matching sport API response within {timeout}s "
                    f"(attempt {attempt+1}/{max_retries})"
                )

        logger.error(
            "[H2] Failed to switch to FSM_MANUAL after "
            f"{max_retries} attempts. LowCmd may be ignored by robot!"
        )
        self._manual_mode_ready = True  # 降级

    # ══════════════════════════════════════════════════════════
    #  内部 — 发布循环
    # ══════════════════════════════════════════════════════════

    def _publish_loop(self):
        rate_delay = 1.0 / self.frequency
        _exception_count = 0
        while rclpy.ok() and not self._shutdown_event.is_set():
            try:
                self.publish_event.wait(timeout=rate_delay)
                self.publish_event.clear()

                # Manual mode 门控：FSM_MANUAL 切换完成前不发布 LowCmd
                if not self._manual_mode_ready:
                    continue

                # ── MoveJ 插值器 ──
                if self._movej_active:
                    elapsed = time.monotonic() - self._movej_start_time
                    if elapsed >= self._movej_duration:
                        with self.publish_lock:
                            self.q_target = self._movej_target_q.copy()
                            self._movej_active = False
                    else:
                        alpha = elapsed / self._movej_duration
                        interp = (self._movej_start_q
                                  + (self._movej_target_q - self._movej_start_q) * alpha)
                        with self.publish_lock:
                            self.q_target = interp

                # dq_limit 管理
                if self._dq_limit_override is not None:
                    self.dq_limit = float(self._dq_limit_override)
                elif self._speed_gradual_max:
                    if self._gradual_start_time is None:
                        self._gradual_start_time = time.perf_counter()
                    t = time.perf_counter() - self._gradual_start_time
                    self.dq_limit = 20.0 + 10.0 * min(1.0, t / 5.0)
                else:
                    self.dq_limit = 20.0

                # 首次命令保护（H2 不做 ArmRequest，MoveJ 期间仍需发布 LowCmd）
                if not self._arm_active:
                    continue

                # ── 电机故障检查 ──
                fault = self._ros_node.fault_buffer.get() if hasattr(self._ros_node, 'fault_buffer') else None
                if fault and fault.get("active", False):
                    if not getattr(self, '_fault_warned', False):
                        logger.warning(
                            f"[MOTOR FAULT] Motor={fault.get('motor_id', -1)} "
                            f"motorstate=0x{fault.get('motorstate', 0):08X} "
                            f"— LowCmd suppressed"
                        )
                        self._fault_warned = True
                    continue
                self._fault_warned = False

                now = time.time()
                elapsed = now - self._last_publish_time
                if elapsed < self._min_publish_interval:
                    time.sleep(self._min_publish_interval - elapsed)

                with self.publish_lock:
                    q_tgt = self.q_target.copy()
                    tau_tgt = self.tauff_target.copy()
                    head_tgt = self.head_target.copy()
                    cur_head = self.last_published_head_q.copy()

                # 以机器人实际位姿为限速基准，防止指令偏离实际导致位置误差超限
                actual_q = self.get_current_dual_arm_q()
                if np.all(actual_q == 0):
                    # state_buffer 尚未就绪时回退到上次下发值
                    with self.publish_lock:
                        actual_q = self.last_published_q.copy()

                # 臂部：矢量等比缩放限速（保持各关节运动比例）
                q_clipped = self._clip_arm_q_target(q_tgt, actual_q, self.dq_limit)
                with self.publish_lock:
                    self.last_published_q = q_clipped.copy()

                _state_data = self._ros_node.state_buffer.get()
                # print(f"[DEBUG] actual_q[0]={actual_q[0]:.4f} "
                #       f"q_tgt[0]={q_tgt[0]:.4f} q_clipped[0]={q_clipped[0]:.4f} "
                #       f"state={'OK' if _state_data else 'EMPTY'} "
                #       f"manual={self._manual_mode_ready} arm={self._arm_active} "
                #       f"fsm={_state_data.mode_machine if _state_data else -1}")

                # 头部限速（直通）
                head_clipped = self._clip_head_q_target(head_tgt, cur_head, self.head_dq_limit)
                with self.publish_lock:
                    self.last_published_head_q = head_clipped.copy()

                # 构建 LowCmd (29 槽位)
                msg = LowCmd()
                msg.mode_pr = 0
                msg.mode_machine = 0
                for i in range(29):
                    kp = float(_H2_KP[i])
                    kd = float(_H2_KD[i])
                    msg.motor_cmd[i] = MotorCmd(
                        mode=1, q=0.0, dq=0.0, tau=0.0,
                        kp=kp, kd=kd, reserve=0)

                # 填充手臂 (slots 15-21, 22-28)
                for i, idx in enumerate(self.left_slots):
                    msg.motor_cmd[idx].q = float(q_clipped[i])
                    msg.motor_cmd[idx].tau = float(tau_tgt[i])
                for i, idx in enumerate(self.right_slots):
                    msg.motor_cmd[idx].q = float(q_clipped[7 + i])
                    msg.motor_cmd[idx].tau = float(tau_tgt[7 + i])

                # 填充头部 (slots 13, 14)
                if self.control_mode.solve_head:
                    try:
                        msg.motor_cmd[self.head_slots[0]].q = float(head_clipped[0])
                        msg.motor_cmd[self.head_slots[1]].q = float(head_clipped[1])
                    except Exception:
                        pass

                # active = []
                # for idx in self.left_slots + self.right_slots + self.head_slots:
                #     q = msg.motor_cmd[idx].q
                #     active.append(f"J{idx}:{q:.4f}")
                # print("Active joints: " + ", ".join(active))

                # CRC + 发布
                self._ros_node.compute_lowcmd_crc(msg)

                self._ros_node.cmd_pub.publish(msg)
                self._last_publish_time = time.time()
                self._write_command_log(q_clipped, head_clipped)

            except Exception:
                _exception_count += 1
                logger.error(
                    f"[_publish_loop] exception #{_exception_count} "
                    f"(rclpy.ok={rclpy.ok()}, shutdown_event={self._shutdown_event.is_set()}):\n"
                    f"{traceback.format_exc()}"
                )
                if not rclpy.ok() or self._shutdown_event.is_set():
                    logger.info("[_publish_loop] shutdown signaled, exiting loop")
                    break
                time.sleep(0.1)

        logger.info(f"[_publish_loop] exited (total exceptions: {_exception_count})")

    def _format_values(self, values):
        return ' '.join(f'{float(v):+.4f}' for v in np.asarray(values, dtype=float).reshape(-1))

    def _write_command_log(self, arm_q, head_q):
        if self._command_log_file is None:
            return
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        milliseconds = int((time.time() % 1.0) * 1000)
        line = (
            f'{timestamp}.{milliseconds:03d} '
            f'arm: {self._format_values(arm_q)}'
        )
        if head_q is not None and head_q.size >= 2:
            line += f'  head: {self._format_values(head_q)}'
        line += '\n'
        try:
            self._command_log_file.write(line)
            self._command_log_file.flush()
        except Exception:
            pass

    def _close_log(self):
        try:
            if self._command_log_file is not None:
                self._command_log_file.close()
                self._command_log_file = None
        except Exception:
            pass

    def _clip_arm_q_target(self, target_q, current_q, velocity_limit):
        """矢量缩放限速，保持各关节运动比例。

        计算所有关节中最大位移与允许位移的比值，
        若超限则整体等比例缩小，轨迹形状不变。
        """
        max_allowed = velocity_limit * self.dt
        delta = target_q - current_q
        max_delta = np.max(np.abs(delta))
        if max_delta <= max_allowed or max_delta < 1e-8:
            return target_q
        scale = max_delta / max_allowed
        return current_q + delta / scale

    def _clip_head_q_target(self, target_q, current_q, velocity_limit):
        max_delta = velocity_limit * self.dt
        delta = target_q - current_q
        return current_q + np.clip(delta, -max_delta, max_delta)

    # ══════════════════════════════════════════════════════════
    #  清理
    # ══════════════════════════════════════════════════════════

    def stop(self):
        """优雅关闭：先通知 publish 线程退出，等线程结束后再销毁 ROS 节点。"""
        logger.info("[stop] signaling shutdown to publish loop...")
        self._shutdown_event.set()
        self.publish_event.set()  # 唤醒可能阻塞在 wait() 的 _publish_loop

        # 等待 publish 线程退出
        if hasattr(self, 'publish_thread') and self.publish_thread is not None and self.publish_thread.is_alive():
            logger.info("[stop] waiting for publish_thread to join...")
            self.publish_thread.join(timeout=3.0)
            if self.publish_thread.is_alive():
                logger.warning("[stop] publish_thread did not exit within 3s")

        # 等待 spin 线程退出
        if hasattr(self, 'spin_thread') and self.spin_thread is not None and self.spin_thread.is_alive():
            logger.info("[stop] waiting for spin_thread to join...")
            self.spin_thread.join(timeout=2.0)

        # 等待 manual 线程退出
        if hasattr(self, '_manual_thread') and self._manual_thread is not None and self._manual_thread.is_alive():
            logger.info("[stop] waiting for manual_thread to join...")
            self._manual_thread.join(timeout=2.0)

        # 关闭日志文件
        self._close_log()

        # 销毁 ROS 节点并 shutdown
        try:
            if rclpy.ok():
                logger.info("[stop] destroying ROS node and shutting down rclpy...")
                self._ros_node.destroy_node()
                rclpy.shutdown()
        except Exception:
            logger.exception("[stop] error during rclpy shutdown")

        logger.info("[stop] complete")
