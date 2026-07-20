"""TOPSTAR_H1 机械臂控制器 — 从旧 topstar_h1_sim_arm_controller.py 迁移。"""
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
from teleop.robot_control.topstar_h1.ros_node import H1RosNode
logger = logging.getLogger(__name__)

# ── KP/KD ───────────────────────────────────────────────────
_H1_KP = [5000.0, 3000.0, 500.0, 500.0, 800.0, 800.0, 800.0, 800.0,
          600.0, 600.0, 600.0, 800.0, 800.0, 800.0, 800.0, 600.0, 600.0, 600.0]
_H1_KD = [100.0, 80.0, 20.0, 20.0, 30.0, 30.0, 30.0, 30.0, 20.0,
          20.0, 20.0, 30.0, 30.0, 30.0, 30.0, 20.0, 20.0, 20.0]


class H1ArmController(BaseArmController):
    """TOPSTAR_H1 机械臂控制器 — 基于 ROS2，事件驱动发布。"""

    def __init__(self, config, control_mode=ControlMode.ARMS_HEAD,
                 frequency=50.0, simulation_mode=True):
        super().__init__(config, control_mode, frequency, simulation_mode)
        logger.info("H1 ArmController: initializing...")

        self.dt = 1.0 / self.frequency
        self.dq_limit = 2.0  # rad/s 默认
        self.head_yaw_limit = (-1.5708, 1.5708)
        self.head_pitch_limit = (-0.6457718, 0.4363323)
        self.head_pitch_ik_limit = (-0.4363323, 0.6457718)
        self.head_dq_limit = 15.0
        self.head_smooth_alpha = 0.8
        self.home_head_q = np.array([0.0, np.deg2rad(-19.0)], dtype=float)

        self.joint_sign_map = np.ones(14, dtype=float)
        if not self.simulation_mode:
            self.joint_sign_map[3] = -1.0
            self.joint_sign_map[5] = -1.0

        self.left_slots = list(range(11, 18))
        self.right_slots = list(range(4, 11))
        self._motor_count = 35

        # 命令锁和目标
        self.publish_lock = threading.Lock()
        self.q_target = np.zeros(14)
        self.tauff_target = np.zeros(14)
        self.last_published_q = np.zeros(14)
        self.last_target_update_ns = 0
        self.last_target_update_wall_ns = 0
        self.last_lowcmd_publish_ns = 0
        self.last_target_to_lowcmd_ms = None
        self._target_sequence = 0
        self._published_sequence = 0
        self.head_target = np.zeros(2)
        self.last_published_head_q = np.zeros(2)
        self.torso_target = np.zeros(2)   # [torso_lift, torso_pitch] (hw convention)
        self.last_published_torso = np.zeros(2)
        self._ee_gripper_state = [0.0, 0.0]
        self._raw_event_sink = None

        # ROS2
        if not rclpy.ok():
            rclpy.init(args=None)
        self._ros_node = H1RosNode()
        logger.info("H1 ArmController: ROS2 node created")

        # 发布控制
        self.publish_event = threading.Event()
        self._shutdown_event = threading.Event()
        self._min_publish_interval = 0.005
        self._last_publish_time = 0.0
        self._joint_hold_until = 0.0
        self._arm_active = False
        self._gradual_start_time = None
        self._dq_limit_override = None
        self._speed_gradual_max = False

        # 线程
        logger.info("H1 ArmController: starting spin thread...")
        self.spin_thread = threading.Thread(target=rclpy.spin, args=(self._ros_node,), daemon=True)
        self.spin_thread.start()
        logger.info(f"H1 ArmController: starting publish loop ({self.frequency} Hz)...")
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

        # 支持 18→14 映射
        if values.size == 18:
            values = np.array([values[s] for s in self.left_slots] +
                              [values[s] for s in self.right_slots], dtype=float)
        if tau.size == 18:
            tau = np.array([tau[s] for s in self.left_slots] +
                           [tau[s] for s in self.right_slots], dtype=float)

        with self.publish_lock:
            self.q_target = values
            self.tauff_target = tau
            self.last_target_update_ns = time.monotonic_ns()
            self.last_target_update_wall_ns = time.time_ns()
            self._target_sequence += 1
        self.publish_event.set()

    def set_raw_event_sink(self, sink) -> None:
        """Attach a non-blocking sink accepting ``(stream_name, event)``."""
        self._raw_event_sink = sink
        self._ros_node.set_state_event_sink(self._emit_lowstate_event if sink else None)

    def _emit_lowstate_event(self, msg, timing) -> None:
        if self._raw_event_sink is None:
            return
        q_hw = np.asarray(
            [msg.motor_state[index].q for index in self.left_slots]
            + [msg.motor_state[index].q for index in self.right_slots],
            dtype=float,
        )
        dq_hw = np.asarray(
            [msg.motor_state[index].dq for index in self.left_slots]
            + [msg.motor_state[index].dq for index in self.right_slots],
            dtype=float,
        )
        self._raw_event_sink('lowstate', {
            **timing,
            'clock_domain': 'robot_source_or_ros_host_receive',
            'q': self._hw_to_sim_arm(q_hw).tolist(),
            'dq': self._hw_to_sim_arm(dq_hw).tolist(),
        })

    def get_current_dual_arm_q(self):
        state = self._ros_node.state_buffer.get()
        if state is None:
            return np.zeros(14)
        res = np.zeros(14)
        for i, idx in enumerate(self.left_slots):
            res[i] = state.motor_state[idx].q
        for i, idx in enumerate(self.right_slots):
            res[7 + i] = state.motor_state[idx].q
        return self._hw_to_sim_arm(res)

    def get_current_dual_arm_state(self):
        """Return q, dq, and the receipt time from one atomic LowState snapshot."""
        state, receive_ns = self._ros_node.state_buffer.get_with_timestamp()
        if state is None:
            return np.zeros(14), np.zeros(14), 0
        q = np.zeros(14)
        dq = np.zeros(14)
        for i, idx in enumerate(self.left_slots):
            q[i] = state.motor_state[idx].q
            dq[i] = state.motor_state[idx].dq
        for i, idx in enumerate(self.right_slots):
            q[7 + i] = state.motor_state[idx].q
            dq[7 + i] = state.motor_state[idx].dq
        return self._hw_to_sim_arm(q), self._hw_to_sim_arm(dq), receive_ns

    def get_current_dual_arm_dq(self):
        state = self._ros_node.state_buffer.get()
        if state is None:
            return np.zeros(14)
        res = np.zeros(14)
        for i, idx in enumerate(self.left_slots):
            res[i] = state.motor_state[idx].dq
        for i, idx in enumerate(self.right_slots):
            res[7 + i] = state.motor_state[idx].dq
        return self._hw_to_sim_arm(res)

    def ctrl_head(self, head_q):
        """控制头部 (IK convention: [yaw, pitch])。"""
        with self.publish_lock:
            h = np.asarray(head_q)
            if h.size >= 2:
                new_target = np.array([
                    float(np.clip(h[0], *self.head_yaw_limit)),
                    float(np.clip(h[1], *self.head_pitch_ik_limit)),
                ], dtype=float)
                self.head_target = (
                    self.head_smooth_alpha * new_target
                    + (1.0 - self.head_smooth_alpha) * self.head_target
                )
        self.publish_event.set()

    def ctrl_torso(self, torso_q):
        """控制躯干 [torso_lift, torso_pitch] (hw convention, 直接写入 LowCmd slot 0,1)。"""
        with self.publish_lock:
            t = np.asarray(torso_q, dtype=float).flatten()
            if t.size >= 2:
                # torso_lift: [0, 0.45] m,  torso_pitch: [0, 1.658] rad
                self.torso_target[0] = float(np.clip(t[0], -0.01, 0.45))
                self.torso_target[1] = float(np.clip(t[1], 0.0, 1.65806279))
            elif t.size == 1:
                # only pitch provided, keep lift unchanged
                self.torso_target[1] = float(np.clip(t[0], 0.0, 1.65806279))
        self.publish_event.set()

    def set_ee_gripper(self, arm_idx, value):
        command_monotonic_ns = time.monotonic_ns()
        command_wall_ns = time.time_ns()
        with self.publish_lock:
            self._ee_gripper_state[arm_idx] = float(value)
            ee_state = list(self._ee_gripper_state)
        self._ros_node.publish_gripper_cmd(arm_idx, float(value))
        if self._raw_event_sink is not None:
            self._raw_event_sink('ee_command', {
                'sequence': int(command_monotonic_ns),
                'publish_monotonic_ns': command_monotonic_ns,
                'publish_wall_ns': command_wall_ns,
                'arm_idx': int(arm_idx),
                'value': float(value),
                'state_right_left': ee_state,
            })
        self.publish_event.set()

    def move_joints_timed(self, joints, duration, head_q=None):
        """MoveJ：通过 /api/arm/request 下发，期间抑制 LowCmd。"""
        self._arm_active = True
        values = [float(v) for v in joints]
        if len(values) == 18:
            request_joints = values
            arm_hw = np.array(
                [request_joints[s] for s in self.left_slots]
                + [request_joints[s] for s in self.right_slots],
                dtype=float,
            )
            arm_values = self._hw_to_sim_arm(arm_hw)
        elif len(values) == 14:
            request_joints = self._sim_to_hw_full_body(values, 18)
            arm_values = np.asarray(values, dtype=float)
        else:
            raise ValueError(f"MoveJ expects 14 or 18 joints, got {len(values)}")

        if head_q is None:
            head_target = self._head_hw_to_ik(request_joints[2:4])
        else:
            head = np.asarray(head_q, dtype=float).reshape(-1)
            if head.size < 2:
                raise ValueError(f"head_q expects yaw and pitch, got {head.size} value(s)")
            head_target = np.array([
                float(np.clip(head[0], *self.head_yaw_limit)),
                float(np.clip(head[1], *self.head_pitch_ik_limit)),
            ])
            request_joints[2:4] = self._head_ik_to_hw(head_target).tolist()

        self._ros_node.publish_movej_request(request_joints, duration)
        self._joint_hold_until = time.monotonic() + duration

        with self.publish_lock:
            self.q_target = arm_values.copy()
            self.tauff_target = np.zeros(14, dtype=float)
            self.last_published_q = arm_values.copy()
            self.head_target = head_target.copy()
            self.last_published_head_q = head_target.copy()
            self.torso_target = np.asarray(request_joints[:2], dtype=float)
            self.last_published_torso = self.torso_target.copy()
        self.publish_event.set()

    def go_home(self, timeout=10.0):
        self.move_joints_timed([
            -1.6, -1.55, 0, 0, 0, 0, 0,
            1.6, -1.571, 0, 0, 0, 0, 0
        ], duration=3.0, head_q=self.home_head_q)

    def wait_for_hold_expire(self, timeout=10.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < self._joint_hold_until:
            if time.monotonic() > deadline:
                break
            time.sleep(0.01)

    def speed_gradual_max(self, t=5.0):
        self._speed_gradual_max = True
        self._gradual_start_time = time.perf_counter()
        self._dq_limit_override = None

    def speed_instant_max(self):
        self._speed_gradual_max = False
        self._dq_limit_override = 30.0
        self.dq_limit = 30.0

    def get_last_published_dual_arm_q(self):
        with self.publish_lock:
            return self.last_published_q.copy()

    def get_head_q(self):
        state = self._ros_node.state_buffer.get()
        if state is None:
            return np.zeros(2)
        try:
            head_hw = np.array([state.motor_state[2].q, state.motor_state[3].q], dtype=float)
            return self._head_hw_to_ik(head_hw)
        except Exception:
            return np.zeros(2)

    def get_current_joint_state(self, mode):
        """H1: ARMS_HEAD 时拼接 14 arm + 2 head = 16 DOF (sim convention)。"""
        arm_q = self.get_current_dual_arm_q()
        if mode == ControlMode.ARMS_HEAD:
            head_q = self.get_head_q()
            return np.concatenate([np.asarray(arm_q, dtype=float)[:14],
                                   np.asarray(head_q, dtype=float)[:2]])
        return np.asarray(arm_q, dtype=float)[:14]

    def connect(self):
        pass  # 初始化时已完成

    def disconnect(self):
        self.stop()

    @property
    def connected(self) -> bool:
        return rclpy.ok()

    # ══════════════════════════════════════════════════════════
    #  内部 — 坐标转换
    # ══════════════════════════════════════════════════════════

    def _sim_to_hw_arm(self, arm_q):
        return np.asarray(arm_q, dtype=float) * self.joint_sign_map

    def _hw_to_sim_arm(self, arm_q):
        return np.asarray(arm_q, dtype=float) * self.joint_sign_map

    def _head_ik_to_hw(self, head_q):
        h = np.asarray(head_q, dtype=float).reshape(-1)
        if h.size < 2:
            return np.zeros(2)
        if self.simulation_mode:
            return np.array([h[0], -h[1]], dtype=float)
        return np.array([-h[0], -h[1]], dtype=float)

    def _head_hw_to_ik(self, head_q):
        h = np.asarray(head_q, dtype=float).reshape(-1)
        if h.size < 2:
            return np.zeros(2)
        if self.simulation_mode:
            return np.array([h[0], -h[1]], dtype=float)
        return np.array([-h[0], -h[1]], dtype=float)

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

                # dq_limit 管理
                if self._dq_limit_override is not None:
                    self.dq_limit = float(self._dq_limit_override)
                elif self._speed_gradual_max:
                    if self._gradual_start_time is None:
                        self._gradual_start_time = time.perf_counter()
                    t = time.perf_counter() - self._gradual_start_time
                    self.dq_limit = 2.0 + 28.0 * min(1.0, t / 5.0)
                else:
                    self.dq_limit = 2.0

                # MoveJ 抑制
                if time.monotonic() < self._joint_hold_until:
                    continue
                if not self._arm_active:
                    continue

                now = time.time()
                elapsed = now - self._last_publish_time
                if elapsed < self._min_publish_interval:
                    time.sleep(self._min_publish_interval - elapsed)

                with self.publish_lock:
                    q_tgt = self.q_target.copy()
                    tau_tgt = self.tauff_target.copy()
                    head_tgt = self.head_target.copy()
                    torso_tgt = self.torso_target.copy()
                    cur_q = self.last_published_q.copy()
                    cur_head = self.last_published_head_q.copy()
                    cur_torso = self.last_published_torso.copy()
                    target_sequence = self._target_sequence
                    target_update_ns = self.last_target_update_ns
                    target_update_wall_ns = self.last_target_update_wall_ns
                    ee_tgt = list(self._ee_gripper_state)

                # 臂部限速 + 坐标转换
                q_clipped = self._clip_arm_q_target(q_tgt, cur_q, self.dq_limit)
                q_hw = self._sim_to_hw_arm(q_clipped)
                tau_hw = self._sim_to_hw_arm(tau_tgt)
                with self.publish_lock:
                    self.last_published_q = q_clipped.copy()

                # 头部限速 + 坐标转换
                head_clipped = self._clip_head_q_target(head_tgt, cur_head, self.head_dq_limit)
                with self.publish_lock:
                    self.last_published_head_q = head_clipped.copy()
                head_hw = self._head_ik_to_hw(head_clipped)

                # 构建 LowCmd
                msg = LowCmd()
                msg.mode_pr = 0
                msg.mode_machine = 0
                for i in range(35):
                    kp = float(_H1_KP[i]) if i < 18 else 0.0
                    kd = float(_H1_KD[i]) if i < 18 else 0.0
                    msg.motor_cmd[i] = MotorCmd(
                        mode=1 if i < 18 else 0, q=0.0, dq=0.0, tau=0.0,
                        kp=kp, kd=kd, reserve=0)

                # 填充手臂 (slots 4-10, 11-17)
                for i, idx in enumerate(self.left_slots):
                    msg.motor_cmd[idx].q = float(q_hw[i])
                    msg.motor_cmd[idx].tau = float(tau_hw[i])
                for i, idx in enumerate(self.right_slots):
                    msg.motor_cmd[idx].q = float(q_hw[7 + i])
                    msg.motor_cmd[idx].tau = float(tau_hw[7 + i])

                # 填充躯干 (slots 0, 1) — 直接写入 hw 值
                torso_clipped = self._clip_torso_target(torso_tgt, cur_torso, 1.0)
                with self.publish_lock:
                    self.last_published_torso = torso_clipped.copy()
                msg.motor_cmd[0].q = float(torso_clipped[0])  # TORSO_LIFT
                msg.motor_cmd[1].q = float(torso_clipped[1])  # TORSO_PITCH

                # 填充头部 (slots 2, 3)。arms_only 时保持 Home 头部姿态。
                msg.motor_cmd[2].q = float(head_hw[0])
                msg.motor_cmd[3].q = float(head_hw[1])

                # ── 调试：打印下发关节角 ────────────────────────────
                # arm_q_debug = []
                # for i, idx in enumerate(self.left_slots):
                #     arm_q_debug.append(f"L{i}:{msg.motor_cmd[idx].q:.4f}")
                # for i, idx in enumerate(self.right_slots):
                #     arm_q_debug.append(f"R{i}:{msg.motor_cmd[idx].q:.4f}")
                # print(
                #     f"[CMD] dq_limit={self.dq_limit:.2f} | "
                #     f"arm=[{', '.join(arm_q_debug)}]"
                # )
                # if self.control_mode.solve_head:
                #     print(
                #         f"[CMD] head=[{msg.motor_cmd[2].q:.4f}, {msg.motor_cmd[3].q:.4f}]"
                #     )

                self._ros_node.compute_lowcmd_crc(msg)
                self._ros_node.cmd_pub.publish(msg)
                self._last_publish_time = time.time()
                publish_wall_ns = time.time_ns()
                with self.publish_lock:
                    publish_ns = time.monotonic_ns()
                    self.last_lowcmd_publish_ns = publish_ns
                    self.last_target_to_lowcmd_ms = (
                        (publish_ns - target_update_ns) / 1e6
                        if target_update_ns > 0 else None
                    )
                    self._published_sequence = target_sequence
                if self._raw_event_sink is not None:
                    self._raw_event_sink('lowcmd', {
                        'sequence': int(target_sequence),
                        'target_update_monotonic_ns': int(target_update_ns),
                        'target_update_wall_ns': int(target_update_wall_ns),
                        'publish_monotonic_ns': int(publish_ns),
                        'publish_wall_ns': int(publish_wall_ns),
                        'q_ik_target': q_tgt.tolist(),
                        'q_commanded': q_clipped.tolist(),
                        'q_commanded_hw': q_hw.tolist(),
                        'ee_command_right_left': ee_tgt,
                        'dq_limit_rad_s': float(self.dq_limit),
                    })

            except Exception:
                _exception_count += 1
                logger.error(
                    f"[_publish_loop] exception #{_exception_count} "
                    f"(rclpy.ok={rclpy.ok()}, shutdown_event={self._shutdown_event.is_set()}):\n"
                    f"{traceback.format_exc()}"
                )
                # 如果 rclpy 已关闭或收到 shutdown 信号，退出循环
                if not rclpy.ok() or self._shutdown_event.is_set():
                    logger.info("[_publish_loop] shutdown signaled, exiting loop")
                    break
                # 短暂休眠避免异常风暴
                time.sleep(0.1)

        logger.info(f"[_publish_loop] exited (total exceptions: {_exception_count})")

    def _clip_arm_q_target(self, target_q, current_q, velocity_limit):
        max_delta = velocity_limit * self.dt
        delta = target_q - current_q
        return current_q + np.clip(delta, -max_delta, max_delta)

    def _clip_head_q_target(self, target_q, current_q, velocity_limit):
        max_delta = velocity_limit * self.dt
        delta = target_q - current_q
        return current_q + np.clip(delta, -max_delta, max_delta)

    def _clip_torso_target(self, target_q, current_q, velocity_limit):
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
        if self.publish_thread is not None and self.publish_thread.is_alive():
            logger.info("[stop] waiting for publish_thread to join...")
            self.publish_thread.join(timeout=3.0)
            if self.publish_thread.is_alive():
                logger.warning("[stop] publish_thread did not exit within 3s")

        # 先 shutdown rclpy，让 spin 线程从 rclpy.spin() 返回，再 join
        if rclpy.ok():
            logger.info("[stop] shutting down rclpy...")
            rclpy.shutdown()

        # 等待 spin 线程退出
        if self.spin_thread is not None and self.spin_thread.is_alive():
            logger.info("[stop] waiting for spin_thread to join...")
            self.spin_thread.join(timeout=2.0)

        # 销毁 ROS 节点
        try:
            logger.info("[stop] destroying ROS node...")
            self._ros_node.destroy_node()
        except Exception:
            logger.exception("[stop] error during destroy_node")

        logger.info("[stop] complete")
