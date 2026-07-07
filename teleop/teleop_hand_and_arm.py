import time
import argparse
import json
from multiprocessing import Value, Array, Lock
import threading
import numpy as np
import logging_mp
logging_mp.basicConfig(level=logging_mp.INFO)
logger_mp = logging_mp.getLogger(__name__)

import os 
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import rclpy

from televuer import TeleVuerWrapper
from teleop.robot_control.factory import create_robot_driver
from teleop.robot_control._base.control_mode import ControlMode
from teleimager.image_client import ImageClient
from teleop.utils.ipc import IPC_Server
from teleop.utils.record import Recorder, RecorderManager
from teleop.utils.lerobot_episode_writer import LeRobotEpisodeWriter, default_repo_id
import queue as _queue
from sshkeyboard import listen_keyboard, stop_listening

# -----------------------------------------------------------------------
# 底盘缓动工具 — 来自 jog GUI (h1_upper_body_jog.py)
# -----------------------------------------------------------------------
def _step_toward(current: float, target: float, accel_step: float, decel_step: float) -> float:
    """每帧向目标值渐进一步，加速/减速步长独立。"""
    if current == target:
        return current
    same_direction = (current == 0.0) or (target == 0.0) or ((current > 0.0) == (target > 0.0))
    speeding_up = same_direction and (abs(target) > abs(current))
    step = accel_step if speeding_up else decel_step
    delta = target - current
    if delta > step:
        return current + step
    if delta < -step:
        return current - step
    return target


def _map_xr_axis_smooth(normalized, deadzone=0.05, output_min=-1.0, output_max=1.0):
    """Map normalised thumbstick value [-1,1] to [output_min, output_max] with dead-zone
    and Hermite smooth curve. Supports asymmetric output ranges (e.g. x_vel: min=-0.6, max=1.0)."""
    if abs(normalized) < deadzone:
        return 0.0
    if normalized > 0:
        t = (normalized - deadzone) / (1.0 - deadzone)
        t = max(0.0, min(1.0, t))
        smooth = 6*t**5 - 15*t**4 + 10*t**3
        return output_max * smooth
    else:
        t = (-normalized - deadzone) / (1.0 - deadzone)
        t = max(0.0, min(1.0, t))
        smooth = 6*t**5 - 15*t**4 + 10*t**3
        return output_min * smooth  # output_min is negative, result in [output_min, 0]




def _require_non_empty_state(name, value):
    """Convert a ROS2 state-like value to a list and fail fast when empty or missing."""
    if value is None:
        raise RuntimeError(f"{name} is None")

    if hasattr(value, "tolist"):
        data = value.tolist()
    else:
        try:
            data = list(value)
        except TypeError:
            data = [value]

    if len(data) == 0:
        raise RuntimeError(f"{name} is empty")

    return data

# state transition
TELEOP_ACTIVE  = False  # Enable to start robot following VR user motion
STOP           = False  # Enable to begin system exit procedure
READY          = False  # Ready to (1) enter TELEOP_ACTIVE state, (2) enter RECORD_RUNNING state
RECORD_RUNNING = False  # True if [Recording]
RECORD_TOGGLE  = False  # Toggle recording state
SERVO_TOGGLE = False  # Toggle ServoJ recording (press 'v' to toggle)

# Teleop mode: "relative_head" (existing) | "relative_pose" (new)
TELEOP_MODE = "relative_head"

# Relative-pose mode: per-arm activation
LEFT_ARM_ACTIVE  = False
RIGHT_ARM_ACTIVE = False

# Right A long-press detection
_right_a_press_start = 0.0
_right_a_long_press_triggered = False

# ── 底盘控制（独立于 TELEOP_ACTIVE，模仿 jog GUI）──
MOTION_ACTIVE = False          # --motion 启动时设为 True
_base_tgt_vx = 0.0
_base_tgt_vy = 0.0
_base_tgt_wz = 0.0
_base_cur_vx = 0.0
_base_cur_vy = 0.0
_base_cur_wz = 0.0
_base_last_t = 0.0            # 上次缓动时间戳

# ── Torso 俯仰（由 control_mode.solve_torso 控制）──
_torso_pitch_target = 0.0      # XR 右摇杆 Y → Torso pitch (hw rad)

# 键盘 WASDQE 按键状态（True=按住）
KEY_W = False
KEY_A = False
KEY_S = False
KEY_D = False
KEY_Q = False
KEY_E = False

# Edge-detection prev states for relative_pose mode buttons
_prev_left_a_relative  = False
_prev_right_a_relative = False
#  -------        ---------                -----------                -----------            ---------
#   state          [Ready]      ==>        [Recording]     ==>         [AutoSave]     -->     [Ready]
#  -------        ---------      |         -----------      |         -----------      |     ---------
#   START           True         |manual      True          |manual      True          |        True
#   READY           True         |set         False         |set         False         |auto    True
#   RECORD_RUNNING  False        |to          True          |to          False         |        False
#                                ∨                          ∨                          ∨
#   RECORD_TOGGLE   False       True          False        True          False                  False
#  -------        ---------                -----------                 -----------            ---------
#  ==> manual: when READY is True, set RECORD_TOGGLE=True to transition.
#  --> auto  : Auto-transition after saving data.

def on_press(key):
    global STOP, TELEOP_ACTIVE, TELEOP_MODE, LEFT_ARM_ACTIVE, RIGHT_ARM_ACTIVE
    global _base_tgt_vx, _base_tgt_vy, _base_tgt_wz, KEY_W, KEY_A, KEY_S, KEY_D, KEY_Q, KEY_E
    # ── 键盘底盘控制（按住时持续移动）──
    # 注意: 'q' 兼顾左转+退出；优先退出（未激活遥操时），激活时左转
    if MOTION_ACTIVE and key not in ('q',):
        if key == 'w':
            KEY_W = True; _update_base_from_keyboard(); return
        elif key == 's':
            KEY_S = True; _update_base_from_keyboard(); return
        elif key == 'a':
            KEY_A = True; _update_base_from_keyboard(); return
        elif key == 'd':
            KEY_D = True; _update_base_from_keyboard(); return
        elif key == 'e':
            KEY_E = True; _update_base_from_keyboard(); return
    # 'q' 的 quit 功能优先；仅当遥操激活时兼作左转
    if key == 'q':
        if TELEOP_ACTIVE:
            if MOTION_ACTIVE:
                KEY_Q = True; _update_base_from_keyboard()
            TELEOP_ACTIVE = False
            logger_mp.info("🟡 Teleop stopped. Press 'q' again to exit.")
        else:
            STOP = True
            logger_mp.info("🔴 Exiting program...")
        return
    if key == 'r':
        if not TELEOP_ACTIVE:
            tv_wrapper.capture_init_head_pose()
            TELEOP_ACTIVE = True
    elif key == 'm':
        if not TELEOP_ACTIVE:
            TELEOP_MODE = "relative_pose" if TELEOP_MODE == "relative_head" else "relative_head"
            if TELEOP_MODE == "relative_pose":
                tv_wrapper.reset_wrist_refs()
                LEFT_ARM_ACTIVE = False
                RIGHT_ARM_ACTIVE = False
                robot.reset_relative_pose_state('both')
            logger_mp.info(f"🔄 Teleop mode switched to: {TELEOP_MODE}")
        else:
            logger_mp.warning("Cannot switch mode while teleop is active. Stop teleop first.")
    elif key == 's' and TELEOP_ACTIVE:
        # Save current session and start a new one
        if 'recorder_manager' in globals() and recorder_manager is not None:
            try:
                recorder_manager.stop_program()
                if 'episode_writer' in globals() and episode_writer is not None:
                    episode_writer.save_episode()
                logger_mp.info("Recording session saved.")
                recorder_manager.ensure_program_active(
                    program_name=args.task_name,
                    description=args.task_desc,
                    filepath=os.path.join(args.task_dir, args.task_name),
                )
                if 'episode_writer' in globals() and episode_writer is not None:
                    episode_writer.create_episode()
                logger_mp.info("New recording session started.")
            except Exception:
                logger_mp.exception("Failed to save/restart recording session")
    elif key == 'v' and TELEOP_ACTIVE:
        # Toggle ServoJ recording
        if 'recorder_manager' in globals() and recorder_manager is not None:
            try:
                if getattr(recorder_manager.recorder, '_servo_recording', False):
                    recorder_manager.stop_servo_recording()
                    if 'episode_writer' in globals() and episode_writer is not None:
                        episode_writer.save_episode()
                    logger_mp.info("ServoJ recording STOPPED")
                else:
                    recorder_manager.ensure_program_active(
                        program_name=args.task_name,
                        description=args.task_desc,
                        filepath=os.path.join(args.task_dir, args.task_name),
                    )
                    recorder_manager.start_servo_recording()
                    if 'episode_writer' in globals() and episode_writer is not None:
                        episode_writer.create_episode()
                    logger_mp.info("ServoJ recording STARTED")
            except Exception:
                logger_mp.exception("Failed to toggle ServoJ recording")
    else:
        logger_mp.warning(f"[on_press] {key} was pressed, but no action is defined for this key.")


def on_release(key):
    """按键释放回调 — 用于键盘底座控制松开即停。"""
    global _base_tgt_vx, _base_tgt_vy, _base_tgt_wz, KEY_W, KEY_A, KEY_S, KEY_D, KEY_Q, KEY_E
    if not MOTION_ACTIVE:
        return
    changed = False
    if key == 'w' and KEY_W:
        KEY_W = False; changed = True
    elif key == 's' and KEY_S:
        KEY_S = False; changed = True
    elif key == 'a' and KEY_A:
        KEY_A = False; changed = True
    elif key == 'd' and KEY_D:
        KEY_D = False; changed = True
    elif key == 'q' and KEY_Q:
        KEY_Q = False; changed = True
    elif key == 'e' and KEY_E:
        KEY_E = False; changed = True
    if changed:
        _update_base_from_keyboard()


def _update_base_from_keyboard():
    """根据当前按键状态计算底盘目标速度（方向 × body_max_speed，对齐 jog GUI）。"""
    global _base_tgt_vx, _base_tgt_vy, _base_tgt_wz
    if not MOTION_ACTIVE:
        _base_tgt_vx = _base_tgt_vy = _base_tgt_wz = 0.0
        return
    vx = 0.0
    if KEY_W:
        vx += 1.0
    if KEY_S:
        vx += -1.0
    vy = 0.0
    if KEY_A:
        vy += 1.0
    if KEY_D:
        vy += -1.0
    wz = 0.0
    if KEY_Q:
        wz += 1.0
    if KEY_E:
        wz += -1.0
    _base_tgt_vx = vx * args.body_max_speed
    _base_tgt_vy = vy * args.body_max_speed
    _base_tgt_wz = wz * args.body_max_speed


def _publish_base_cmd(tele_data=None):
    """发布底盘 Twist 指令 — 独立于 TELEOP_ACTIVE，受 --motion 控制。

    设计模仿 jog GUI：
      1. 键盘/XR 设定目标速度（_base_tgt_*）
      2. 每帧 _step_toward 从当前速度向目标速度缓动
      3. 发布缓动后的速度到 /base_cmd
    """
    global _base_tgt_vx, _base_tgt_vy, _base_tgt_wz
    global _base_cur_vx, _base_cur_vy, _base_cur_wz, _base_last_t

    if not MOTION_ACTIVE:
        return
    try:
        arm_ctrl
    except NameError:
        return

    now = time.monotonic()
    dt = now - _base_last_t if _base_last_t > 0 else (1.0 / args.frequency)
    _base_last_t = now
    dt = max(0.001, min(dt, 0.1))

    # ── XR 摇杆 → 覆盖键盘目标（仅方向，幅值由 body_max_speed 缩放）──
    if tele_data is not None and args.input_mode == "controller":
        try:
            has_ls = getattr(tele_data, 'left_ctrl_thumbstick', False)
            has_rs = getattr(tele_data, 'right_ctrl_thumbstick', False)
        except Exception:
            has_ls = has_rs = False
        if has_ls or has_rs:
            xr_vx = 0.0; xr_vy = 0.0; xr_wz = 0.0
            if has_ls:
                ls = getattr(tele_data, 'left_ctrl_thumbstickValue', [0.0, 0.0])
                # _map_xr_axis_smooth 返回 [-1, 1] 方向，乘以 body_max_speed 得幅值
                xr_vx = _map_xr_axis_smooth(-ls[1], 0.05, -1.0, 1.0) * args.body_max_speed
                xr_vy = _map_xr_axis_smooth(-ls[0], 0.05, -1.0, 1.0) * args.body_max_speed
            if has_rs:
                rs = getattr(tele_data, 'right_ctrl_thumbstickValue', [0.0, 0.0])
                xr_wz = _map_xr_axis_smooth(-rs[0], 0.05, -1.0, 1.0) * args.body_max_speed
            if abs(xr_vx) > 0.01 or abs(xr_vy) > 0.01 or abs(xr_wz) > 0.01:
                _base_tgt_vx, _base_tgt_vy, _base_tgt_wz = xr_vx, xr_vy, xr_wz

    # ── _step_toward 缓动（硬编码常量，对齐 jog GUI）──
    _LIN_ACCEL = 0.8; _LIN_DECEL = 1.2
    _ANG_ACCEL = 1.5; _ANG_DECEL = 2.0
    lin_accel_step = _LIN_ACCEL * dt
    lin_decel_step = _LIN_DECEL * dt
    ang_accel_step = _ANG_ACCEL * dt
    ang_decel_step = _ANG_DECEL * dt
    _base_cur_vx = _step_toward(_base_cur_vx, _base_tgt_vx, lin_accel_step, lin_decel_step)
    _base_cur_vy = _step_toward(_base_cur_vy, _base_tgt_vy, lin_accel_step, lin_decel_step)
    _base_cur_wz = _step_toward(_base_cur_wz, _base_tgt_wz, ang_accel_step, ang_decel_step)

    # ── 发布 Twist ──
    if hasattr(arm_ctrl, '_ros_node') and hasattr(arm_ctrl._ros_node, 'publish_base_cmd'):
        arm_ctrl._ros_node.publish_base_cmd(_base_cur_vx, _base_cur_vy, _base_cur_wz)


def _publish_torso_cmd(tele_data=None):
    """发布 Torso 俯仰到 LowCmd slot 1 — 仅当 control_mode.solve_torso=True 时激活。

    右摇杆 Y 轴 → torso_pitch，经 arm_ctrl.ctrl_torso() → _publish_loop 限速后写入 LowCmd。
    """
    global _torso_pitch_target
    if not control_mode.solve_torso:
        return
    try:
        arm_ctrl
    except NameError:
        return
    if not hasattr(arm_ctrl, 'ctrl_torso'):
        return

    if tele_data is not None and args.input_mode == "controller":
        try:
            has_rs = getattr(tele_data, 'right_ctrl_thumbstick', False)
        except Exception:
            has_rs = False
        if has_rs:
            rs = getattr(tele_data, 'right_ctrl_thumbstickValue', [0.0, 0.0])
            # _map_xr_axis_smooth 返回 [-1, 1] 方向，映射到 torso_pitch_max
            pitch = _map_xr_axis_smooth(-rs[1], 0.05, -1.65806279, 1.65806279)
            _torso_pitch_target = pitch if abs(pitch) > 0.001 else 0.0
        else:
            _torso_pitch_target = 0.0
    else:
        _torso_pitch_target = 0.0

    arm_ctrl.ctrl_torso([0.0, _torso_pitch_target])


def get_state() -> dict:
    """Return current heartbeat state"""
    global TELEOP_ACTIVE, STOP, READY, TELEOP_MODE, LEFT_ARM_ACTIVE, RIGHT_ARM_ACTIVE
    is_servo = False
    if 'recorder_manager' in globals() and recorder_manager is not None:
        is_servo = getattr(recorder_manager.recorder, '_servo_recording', False)
    return {
        "START": TELEOP_ACTIVE,
        "STOP": STOP,
        "READY": READY,
        "RECORD_RUNNING": is_servo,
        "TELEOP_MODE": TELEOP_MODE,
        "LEFT_ARM_ACTIVE": LEFT_ARM_ACTIVE,
        "RIGHT_ARM_ACTIVE": RIGHT_ARM_ACTIVE,
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # basic control parameters
    parser.add_argument('--frequency', type = float, default = 30.0, help = 'control and record \'s frequency')
    parser.add_argument('--input-mode', type=str, choices=['hand', 'controller'], default='hand', help='Select XR device input tracking source')
    parser.add_argument('--display-mode', type=str, choices=['immersive', 'ego', 'pass-through'], default='immersive', help='Select XR device display mode')
    parser.add_argument('--robot', type=str, default='TOPSTAR_H1', help='Robot model (e.g. TOPSTAR_H1, TOPSTAR_H2)')
    parser.add_argument('--control-mode', type=str, default='arms_head',
                        choices=['arms_only', 'arms_head', 'arms_head_torso', 'full_body'],
                        help='Control mode')
    parser.add_argument('--ee', type=str, choices=['dex1', 'dex3', 'inspire_ftp', 'inspire_dfx', 'brainco', 'suction_cup'], help='Select end effector controller')
    parser.add_argument('--img-server-ip', type=str, default='192.168.123.164', help='IP address of image server, used by teleimager and televuer')
    parser.add_argument('--network-interface', type=str, default=None, help='Network interface for dds communication, e.g., eth0, wlan0. If None, use default interface.')
    # mode flags
    parser.add_argument('--motion', action='store_true',
                        help='Enable chassis / base movement control (keyboard WASDQE or XR controller thumbsticks)')
    parser.add_argument('--headless', action='store_true', help='Enable headless mode (no display)')
    parser.add_argument('--sim', action='store_true', help='Enable simulation mode (passed to RobotDriver, does NOT control body)')
    parser.add_argument('--ipc', action = 'store_true', help = 'Enable IPC server to handle input; otherwise enable sshkeyboard')
    parser.add_argument('--affinity', action = 'store_true', help = 'Enable high priority and set CPU affinity mode')
    # replay options
    parser.add_argument('--replay', nargs='?', const='fast', default=None, help='Enable replay: --replay (fast) or --replay first (slow/safe)')
    parser.add_argument('--replay-file', dest='replay_file', type=str, default=None, help='Path to trajectory JSON file for replay')
    # record mode and task info
    parser.add_argument('--record', action = 'store_true', help = 'Enable data recording mode')
    parser.add_argument('--task-dir', type = str, default = './utils/data/', help = 'path to save data')
    parser.add_argument('--task-name', type = str, default = 'pick cube', help = 'task file name for recording')
    parser.add_argument('--task-goal', type = str, default = 'pick up cube.', help = 'task goal for recording at json file')
    parser.add_argument('--task-desc', type = str, default = 'task description', help = 'task description for recording at json file')
    parser.add_argument('--task-steps', type = str, default = 'step1: do this; step2: do that;', help = 'task steps for recording metadata')
    parser.add_argument('--lerobot-repo-id', type=str, default=None, help='LeRobot dataset repo id metadata, defaults to local/<task-name>')
    parser.add_argument('--sync-tolerance-s', type=float, default=1e-4,
                        help='LeRobot-style timestamp tolerance metadata for recorded samples')
    parser.add_argument('--camera-sync-tolerance-s', type=float, default=0.02,
                        help='Maximum allowed software timestamp skew between camera frames')
    # ── Body / Chassis 运动参数（依赖 --motion，对齐 jog GUI）──
    parser.add_argument('--body-max-speed',         type=float, default=1.0,  help='Global speed scaling factor (0~1), like jog GUI slider')
    # ── 其他 ──
    parser.add_argument('--arm-scale', type=float, default=1.0, help='Arm reach scaling factor (e.g., 0.8)')

    args = parser.parse_args()
    logger_mp.info(f"args: {args}")

    try:
        # ── 初始化 RobotDriver ──
        control_mode = ControlMode.from_str(args.control_mode)
        logger_mp.info(f"Robot: {args.robot}, ControlMode: {control_mode.value}, EE: {args.ee}")
        logger_mp.info(f"Initializing robot driver: {args.robot} (sim={args.sim})...")
        robot = create_robot_driver(
            args.robot,
            control_mode=control_mode,
            frequency=args.frequency,
            simulation_mode=args.sim,
            arm_scale=args.arm_scale,
            ee_type=args.ee,
            verbose=False,
        )
        logger_mp.info(f"Robot driver ready: {robot.config.model_name}, "
                       f"mode={control_mode.value}, "
                       f"controller={type(robot.controller).__name__}")
        # 为后向兼容保留 arm_ctrl 引用
        arm_ctrl = robot.controller
        arm_ik = robot.ik

        # ── DDS / IPC / keyboard ──
        if args.ipc:
            ipc_server = IPC_Server(on_press=on_press, get_state=get_state)
            ipc_server.start()
            logger_mp.info("Input source: IPC server")
        else:
            listen_keyboard_thread = threading.Thread(
                target=listen_keyboard,
                kwargs={"on_press": on_press, "on_release": on_release,
                         "until": None, "sequential": False},
                daemon=True,
            )
            listen_keyboard_thread.start()
            logger_mp.info("Input source: keyboard")

        # ── image client（headless 模式跳过，使用默认配置）──
        img_client = None
        head_img = None
        left_wrist_img = None
        right_wrist_img = None
        if args.headless:
            camera_config = {
                'head_camera': {
                    'binocular': False,
                    'image_shape': (480, 640),
                    'enable_zmq': False,
                    'enable_webrtc': False,
                    'webrtc_port': 0,
                },
                'left_wrist_camera': {'enable_zmq': False},
                'right_wrist_camera': {'enable_zmq': False},
            }
            logger_mp.info("🖥️  Headless mode — no image server, using defaults")
            args.display_mode = 'pass-through'  # headless 无需沉浸模式
        else:
            img_client = ImageClient(host=args.img_server_ip, request_bgr=True)
            camera_config = img_client.get_cam_config()
            logger_mp.debug(f"Camera config: {camera_config}")
        xr_need_local_img = not (args.display_mode == 'pass-through' or camera_config['head_camera']['enable_webrtc'])

        # ── televuer_wrapper ──
        tv_wrapper = TeleVuerWrapper(
            use_hand_tracking=args.input_mode == "hand",
            binocular=camera_config['head_camera']['binocular'],
            img_shape=camera_config['head_camera']['image_shape'],
            display_mode=args.display_mode,
            zmq=camera_config['head_camera']['enable_zmq'],
            webrtc=camera_config['head_camera']['enable_webrtc'],
            webrtc_url=f"https://{args.img_server_ip}:{camera_config['head_camera']['webrtc_port']}/offer",
        )

        # ── Vuer 子进程状态检查 ──
        time.sleep(1.5)  # 等待 Vuer 子进程初始化
        if hasattr(tv_wrapper, 'tvuer') and hasattr(tv_wrapper.tvuer, 'process'):
            _alive = tv_wrapper.tvuer.process.is_alive()
            if _alive:
                logger_mp.info(f"✅ Vuer subprocess RUNNING (pid={tv_wrapper.tvuer.process.pid})")
            else:
                logger_mp.error(
                    "❌ Vuer subprocess is DEAD — WebSocket connection impossible!\n"
                    "   Check: ① SSL cert/key in ~/.config/xr_teleoperate/ or $XR_TELEOP_CERT\n"
                    "          ② Port 8012 not occupied by other process\n"
                    "          ③ Vuer startup exceptions (run with --headless to see stderr)"
                )

        # affinity mode (if you dont know what it is, then you probably don't need it)
        if args.affinity:
            import psutil
            p = psutil.Process(os.getpid())
            p.cpu_affinity([0,1,2,3]) # Set CPU affinity to cores 0-3
            try:
                p.nice(-20)           # Set highest priority
                logger_mp.info("Set high priority successfully.")
            except psutil.AccessDenied:
                logger_mp.warning("Failed to set high priority. Please run as root.")
                
            for child in p.children(recursive=True):
                try:
                    logger_mp.info(f"Child process {child.pid} name: {child.name()}")
                    child.cpu_affinity([5,6])
                    child.nice(-20)
                except psutil.AccessDenied:
                    pass

        # ── Motion 模式初始化 ──
        MOTION_ACTIVE = args.motion
        if args.motion:
            _base_last_t = time.monotonic()

        # record + headless / non-headless mode
        if args.record:
            # trajectory recorder + manager: samples arm joint states and writes trajectory JSON
            try:
                recorder = Recorder(buffer_size=1000)

                # Build EE action getter for suction_cup
                ee_action_getter = None
                if args.ee == "suction_cup" and hasattr(arm_ctrl, '_ee_gripper_state'):
                    def _get_suction_cup_ee_action():
                        state = getattr(arm_ctrl, '_ee_gripper_state', [0.0, 0.0])
                        return list(state) if state else [0.0, 0.0]
                    ee_action_getter = _get_suction_cup_ee_action

                recorder_manager = RecorderManager(
                    recorder, arm_ctrl,
                    ik=arm_ik if 'arm_ik' in globals() else None,
                    frequency=args.frequency,
                    ee_type=args.ee,
                    ee_action_getter=ee_action_getter,
                    control_mode=control_mode,
                    robot_model=robot.config.model_name,
                )
                recorder_manager.start()
            except Exception:
                logger_mp.exception('Failed to initialize trajectory recorder')

            # Multi-modal dataset writer: samples RGB/depth images + states/actions
            # directly into a LeRobotDataset. Alignment timing details are stored
            # as a sidecar under meta/tele_robot_alignment.jsonl.
            try:
                lerobot_root = os.path.join(args.task_dir, args.task_name)
                episode_writer = LeRobotEpisodeWriter(
                    root=lerobot_root,
                    repo_id=args.lerobot_repo_id or default_repo_id(args.task_name),
                    fps=args.frequency,
                    task=args.task_goal or args.task_name,
                    robot_type=robot.config.model_name,
                    tolerance_s=args.sync_tolerance_s,
                    use_videos=True,
                )
            except Exception:
                logger_mp.exception('Failed to initialize LeRobot dataset writer')

        # replay mode: if requested, set up Replay + adapter + consumer and run replay then exit
        if args.replay is not None:
            if not args.replay_file:
                logger_mp.error("--replay requires --replay-file to be set")
                sys.exit(2)
            try:
                from teleop.utils.replay import (Replay, ReplayOptions,
                                                 ReplayConsumer,
                                                 get_replay_queue,
                                                 replay_executor)

                player = Replay(buffer_size=1000, replay_executor=replay_executor)
                trajectory_id, summary = player.load_replay(
                    args.replay_file, verify_signature=False
                )
                playback_hz = float(summary.frequency)
                if playback_hz <= 0:
                    raise ValueError(
                        "Replay JSON metadata.frequency must be greater than 0"
                    )

                # ── EE 校验 ──
                with open(args.replay_file, 'r') as _f:
                    _traj_data = json.load(_f)
                _traj_meta = _traj_data.get("metadata", {})
                _traj_ee = _traj_meta.get("ee")
                _traj_ctrl = _traj_meta.get("control_mode")

                if _traj_ee and not args.ee:
                    logger_mp.error(
                        f"轨迹要求 EE='{_traj_ee}'，但未指定 --ee 参数"
                    )
                    sys.exit(2)
                if not _traj_ee and args.ee:
                    logger_mp.error(
                        f"轨迹未记录 EE 信息，但指定了 --ee={args.ee}"
                    )
                    sys.exit(2)
                if _traj_ee and args.ee and _traj_ee != args.ee:
                    logger_mp.error(
                        f"EE 不匹配: 轨迹='{_traj_ee}'，CLI='{args.ee}'"
                    )
                    sys.exit(2)

                # ── ControlMode 校验 ──
                replay_mode = control_mode
                if _traj_ctrl:
                    traj_mode = ControlMode.from_str(_traj_ctrl)
                    if replay_mode.ik_dof > traj_mode.ik_dof:
                        logger_mp.error(
                            f"回放模式 {replay_mode.value} 需要 {replay_mode.ik_dof} DOF，"
                            f"但轨迹只有 {traj_mode.value} ({traj_mode.ik_dof} DOF)"
                        )
                        sys.exit(2)
                    logger_mp.info(
                        f"轨迹模式={traj_mode.value}, 回放模式={replay_mode.value}"
                    )
                else:
                    logger_mp.info(
                        "旧格式轨迹（无 control_mode），按 arms_only 处理"
                    )

                session_id = player.prepare_replay(trajectory_id, ReplayOptions())

                # ── pre-replay: go home, then move to first trajectory point ──
                logger_mp.info("Moving to home pose...")
                robot.go_home()
                if hasattr(arm_ctrl, 'wait_for_hold_expire'):
                    arm_ctrl.wait_for_hold_expire()
                logger_mp.info("Home pose reached.")

                _first_points = _traj_data.get("points", [])
                if _first_points:
                    _first = _first_points[0]
                    if _first.get("joint_pose") is not None:
                        _first_jp_deg = [float(v) for v in _first["joint_pose"]]
                        _jp_summary = (
                            '[' + ', '.join(f'{v:.1f}°'
                                            for v in _first_jp_deg[:3])
                            + ' ...]'
                        )
                        logger_mp.info(
                            f"📍 First point joint_pose[0:3] = {_jp_summary}"
                        )
                        _first_duration = float(
                            _first.get("duration",
                                       _first.get("move_time", 3.0))
                        )
                        if _first_duration <= 0:
                            _first_duration = 3.0
                        logger_mp.info(
                            f"⏳ Moving to first point ({_first_duration:.1f}s)..."
                        )
                        robot.replay_point(_first, replay_mode)
                        if hasattr(arm_ctrl, 'wait_for_hold_expire'):
                            arm_ctrl.wait_for_hold_expire(
                                timeout=_first_duration + 1.0
                            )
                        logger_mp.info("✅ First trajectory point reached.")
                else:
                    logger_mp.warning("⚠️  Trajectory has no points!")

                # ── replay speed control ──
                if args.replay == 'first':
                    logger_mp.info(
                        "🐢 Replay mode: first run — dq_limit = 2 rad/s (slow, safe)"
                    )
                else:
                    arm_ctrl.speed_instant_max()
                    logger_mp.info("⚡ Replay mode: fast — dq_limit = 30 rad/s")

                # ── Stage 3/4: Execute trajectory replay ──
                logger_mp.info("")
                logger_mp.info("╔══════════════════════════════════════╗")
                logger_mp.info("║  Stage 3/4: Executing trajectory     ║")
                logger_mp.info("╚══════════════════════════════════════╝")
                _replay_progress = [0, summary.total_points or 0]

                def _dispatcher(point: dict):
                    t = point.get('type')
                    pid = point.get('id', '?')
                    try:
                        robot.replay_point(point, replay_mode)
                        logger_mp.debug(f"Dispatched {t} id={pid}")
                    except Exception:
                        logger_mp.exception(
                            f"Failed to dispatch {t} id={pid}"
                        )

                    _replay_progress[0] += 1
                    if (_replay_progress[0]
                            % max(1, _replay_progress[1] // 10) == 0
                            or _replay_progress[0] <= 1):
                        pct = (_replay_progress[0]
                               / max(1, _replay_progress[1]) * 100)
                        logger_mp.info(
                            f"  ▶ replay progress: "
                            f"{_replay_progress[0]}/{_replay_progress[1]} "
                            f"({pct:.0f}%)  |  id={pid}  type={t}"
                        )

                replay_queue = get_replay_queue()
                consumer = ReplayConsumer(
                    replay_queue, _dispatcher, frequency=playback_hz
                )
                consumer.start()

                player.start_replay(session_id)

                # wait until replay finishes, with timeout fallback
                _replay_timeout = max(
                    30.0,
                    (_replay_progress[1] / max(1.0, playback_hz)) * 3.0,
                )
                _replay_deadline = time.monotonic() + _replay_timeout
                _last_progress = 0
                _stall_deadline = time.monotonic() + 5.0
                while True:
                    status = player.get_replay_status(session_id)
                    if status.state.value in (
                            "Finished", "Stopped", "Error"):
                        logger_mp.info(
                            f"Replay status: {status.state.value}"
                        )
                        break
                    # detect completion by progress stalling
                    if (_replay_progress[0] >= _replay_progress[1]
                            and _replay_progress[1] > 0):
                        if _replay_progress[0] > _last_progress:
                            _last_progress = _replay_progress[0]
                            _stall_deadline = time.monotonic() + 2.0
                        elif time.monotonic() > _stall_deadline:
                            logger_mp.info(
                                f"Replay all points dispatched "
                                f"({_replay_progress[0]}/"
                                f"{_replay_progress[1]}), proceeding..."
                            )
                            break
                    # absolute timeout
                    if time.monotonic() > _replay_deadline:
                        logger_mp.warning(
                            f"Replay timeout ({_replay_timeout:.0f}s), "
                            f"proceeding with {_replay_progress[0]}/"
                            f"{_replay_progress[1]} points..."
                        )
                        break
                    time.sleep(0.05)

                consumer.stop()
                logger_mp.info(
                    f"✅ Trajectory replay complete.  "
                    f"({_replay_progress[0]} points executed)"
                )

                # ── Stage 4/4: Wait for Enter, then return to home ──
                logger_mp.info("")
                logger_mp.info("╔══════════════════════════════════════╗")
                logger_mp.info("║  Stage 4/4: Return to HOME           ║")
                logger_mp.info("╚══════════════════════════════════════╝")
                logger_mp.info("")
                logger_mp.info("🔴 Press [Enter] to return to home pose and exit...")
                print("\n" + "=" * 50)
                print("  🔴 按 [Enter] 键回到初始姿态并退出")
                print("=" * 50 + "\n")
                try:
                    input()
                except (EOFError, KeyboardInterrupt):
                    pass

                logger_mp.info("Returning to home pose...")
                robot.go_home()
                if hasattr(arm_ctrl, 'wait_for_hold_expire'):
                    arm_ctrl.wait_for_hold_expire()
                logger_mp.info("Home pose reached. Exiting.")
                sys.exit(0)
            except Exception:
                logger_mp.error(
                    "Replay failed:\n"
                    + __import__('traceback').format_exc()
                )
                sys.exit(1)

        logger_mp.info("----------------------------------------------------------------")
        logger_mp.info("🟢  Press [r] to start syncing the robot with your movements.")
        if args.record:
            logger_mp.info("🟡  Press [s] to START or SAVE recording (toggle cycle).")
        else:
            logger_mp.info("🔵  Recording is DISABLED (run with --record to enable).")
        logger_mp.info("🔴  Press [q] to stop and exit the program.")
        logger_mp.info("⚠️  IMPORTANT: Please keep your distance and stay safe.")
        READY = True                  # now ready to (1) enter TELEOP_ACTIVE state

        # 启动时回零一次，后续进入/退出遥操保持当前姿态
        arm_ctrl.go_home()

        while not STOP: # wait for start or stop signal.
            logger_mp.info("🔵 Waiting for teleop start (press r or left A)...")
            while not TELEOP_ACTIVE and not STOP:
                time.sleep(0.033)
                if img_client is not None and camera_config['head_camera']['enable_zmq'] and xr_need_local_img:
                    head_img = img_client.get_head_frame()
                    tv_wrapper.render_to_xr(head_img)
                try:
                    tele_data = tv_wrapper.get_tele_data()

                    # ── Right A: 统一 press 跟踪 + 短按/长按检测 ──
                    #  短按: release < 3s → arm toggle (relative_pose) / enter teleop (relative_head)
                    #  长按: hold ≥ 3s    → mode switch
                    _right_a_short = False
                    if hasattr(tele_data, 'right_ctrl_aButton') and tele_data.right_ctrl_aButton:
                        _now = time.time()
                        if not getattr(tv_wrapper, '_prev_right_ctrl_aButton', False):
                            _right_a_press_start = _now
                        elif _now - _right_a_press_start >= 3.0 and not _right_a_long_press_triggered:
                            _right_a_long_press_triggered = True
                            # Long press: mode switch
                            TELEOP_MODE = "relative_pose" if TELEOP_MODE == "relative_head" else "relative_head"
                            if TELEOP_MODE == "relative_pose":
                                tv_wrapper.reset_wrist_refs()
                                LEFT_ARM_ACTIVE = False
                                RIGHT_ARM_ACTIVE = False
                                robot.reset_relative_pose_state('both')
                            logger_mp.info(f"🔄 Right A long press: Teleop mode switched to: {TELEOP_MODE}")
                        tv_wrapper._prev_right_ctrl_aButton = True
                    else:
                        # Released
                        if getattr(tv_wrapper, '_prev_right_ctrl_aButton', False):
                            if not _right_a_long_press_triggered and _right_a_press_start > 0:
                                _right_a_short = True  # Released before 3s → short press
                        _right_a_press_start = 0.0
                        _right_a_long_press_triggered = False
                        tv_wrapper._prev_right_ctrl_aButton = False

                    if TELEOP_MODE == "relative_head":
                        # ── relative_head: Left A toggles TELEOP_ACTIVE ──
                        if hasattr(tele_data, 'left_ctrl_aButton') and tele_data.left_ctrl_aButton:
                            if not getattr(tv_wrapper, '_prev_left_ctrl_aButton', False):
                                try:
                                    tv_wrapper.capture_init_head_pose()
                                except Exception:
                                    pass
                                TELEOP_ACTIVE = True
                                logger_mp.info("XR Left A/X pressed in waiting loop -> TELEOP_ACTIVE=True")
                            tv_wrapper._prev_left_ctrl_aButton = True
                        else:
                            tv_wrapper._prev_left_ctrl_aButton = False
                    else:  # relative_pose
                        # ── relative_pose: Left A activates left arm ──
                        if hasattr(tele_data, 'left_ctrl_aButton') and tele_data.left_ctrl_aButton:
                            if not getattr(tv_wrapper, '_prev_left_ctrl_aButton', False):
                                try:
                                    tv_wrapper.capture_left_wrist_ref()
                                except Exception:
                                    pass
                                LEFT_ARM_ACTIVE = True
                                TELEOP_ACTIVE = True
                                robot.reset_relative_pose_state('right')
                                logger_mp.info("XR Left A/X: Left hand → Right arm activated")
                            tv_wrapper._prev_left_ctrl_aButton = True
                        else:
                            tv_wrapper._prev_left_ctrl_aButton = False

                        # ── relative_pose: Right A short press (release < 3s) activates right arm ──
                        if _right_a_short:
                            try:
                                tv_wrapper.capture_right_wrist_ref()
                            except Exception:
                                pass
                            RIGHT_ARM_ACTIVE = True
                            TELEOP_ACTIVE = True
                            robot.reset_relative_pose_state('left')
                            logger_mp.info("XR Right A: Right hand → Left arm activated")

                    # ── 等待循环中支持录制控制 ──
                    # Left B/Y: 切换 ServoJ 连续轨迹录制
                    if args.record and hasattr(tele_data, 'left_ctrl_bButton') and tele_data.left_ctrl_bButton:
                        if not getattr(tv_wrapper, '_prev_left_ctrl_bButton', False):
                            try:
                                if getattr(recorder_manager.recorder, '_servo_recording', False):
                                    recorder_manager.stop_servo_recording()
                                    if 'episode_writer' in globals() and episode_writer is not None:
                                        episode_writer.save_episode()
                                    logger_mp.info("XR Left B/Y: ServoJ recording STOPPED")
                                else:
                                    recorder_manager.ensure_program_active(
                                        program_name=args.task_name,
                                        description=args.task_desc,
                                        filepath=os.path.join(args.task_dir, args.task_name),
                                    )
                                    recorder_manager.start_servo_recording()
                                    if 'episode_writer' in globals() and episode_writer is not None:
                                        episode_writer.create_episode()
                                    logger_mp.info("XR Left B/Y: ServoJ recording STARTED")
                            except Exception:
                                logger_mp.exception('Failed to toggle ServoJ via XR')
                        tv_wrapper._prev_left_ctrl_bButton = True
                    else:
                        tv_wrapper._prev_left_ctrl_bButton = False

                    # Right B/Y: 记录单个 MoveJ 点
                    if args.record and hasattr(tele_data, 'right_ctrl_bButton') and tele_data.right_ctrl_bButton:
                        if not getattr(tv_wrapper, '_prev_right_ctrl_bButton', False):
                            try:
                                recorder_manager.record_movej_point()
                                _pt_count = getattr(recorder_manager.recorder, 'total_points', 0)
                                _q = arm_ctrl.get_current_dual_arm_q()
                                _q_summary = '[' + ', '.join(f'{v:+.2f}' for v in _q[:3]) + ' ...]'
                                logger_mp.info(f"🎯 XR Right B/Y: MoveJ point #{_pt_count} recorded  |  joints[0:3]={_q_summary}")
                            except Exception:
                                logger_mp.exception('Failed to record MoveJ point via XR')
                        tv_wrapper._prev_right_ctrl_bButton = True
                    else:
                        tv_wrapper._prev_right_ctrl_bButton = False

                    # ── 等待循环中响应 trigger/squeeze ──
                    if hasattr(robot, 'handler_registry') and robot.handler_registry.count > 0:
                        swap = TELEOP_MODE == "relative_pose"
                        robot._prev_ee_state = robot.handler_registry.dispatch_trigger_squeeze(
                            tele_data, robot._prev_ee_state, swap_sides=swap,
                        )

                    # ── 等待循环中持续发布底盘 & Torso 指令（独立于 TELEOP_ACTIVE）──
                    _publish_base_cmd(tele_data)
                    _publish_torso_cmd(tele_data)

                except Exception:
                    pass
                # 打印等待循环中的 TELEOP_ACTIVE/STOP 状态，便于调试
                # logger_mp.info(f"Waiting loop state: TELEOP_ACTIVE={TELEOP_ACTIVE}, STOP={STOP}")

            if STOP:
                break
            
            logger_mp.info("---------------------🚀start Tracking🚀-------------------------")
            robot.speed_gradual_max()
            # main loop. robot start to follow VR user's motion
            while TELEOP_ACTIVE and not STOP:
                start_time = time.time()
                # get image
                if camera_config['head_camera']['enable_zmq']:
                    if (args.record or xr_need_local_img) and img_client is not None:
                        head_img = img_client.get_head_frame()
                    if xr_need_local_img:
                        tv_wrapper.render_to_xr(head_img)
                if args.record and img_client is not None:
                    if camera_config.get('left_wrist_camera', {}).get('enable_zmq', False):
                        left_wrist_img = img_client.get_left_wrist_frame()
                    if camera_config.get('right_wrist_camera', {}).get('enable_zmq', False):
                        right_wrist_img = img_client.get_right_wrist_frame()

                # record mode
                if args.record and RECORD_TOGGLE:
                    RECORD_TOGGLE = False
                    if not RECORD_RUNNING:
                        RECORD_RUNNING = True
                        try:
                            # start trajectory recording program
                            recorder_manager.start_program(args.task_name, args.task_desc, filepath=os.path.join(args.task_dir, args.task_name), frequency=args.frequency)
                        except Exception:
                            logger_mp.exception('Failed to start trajectory recording program')
                    else:
                        RECORD_RUNNING = False
                        try:
                            recorder_manager.stop_program()
                            episode_path = getattr(recorder, 'current_file', None)
                            if episode_path is not None:
                                logger_mp.info(f"⏹️ Stopping Recording... Trajectory saved to: {episode_path}")
                        except Exception:
                            logger_mp.exception('Failed to stop trajectory recording program')
                        if args.sim:
                            pass  # sim reset TODO

                # servo toggle (press 'v' to toggle ServoJ recording when recording)
                if args.record and SERVO_TOGGLE:
                    SERVO_TOGGLE = False
                    if RECORD_RUNNING:
                        try:
                            if getattr(recorder_manager.recorder, '_servo_recording', False):
                                recorder_manager.stop_servo_recording()
                            else:
                                recorder_manager.start_servo_recording()
                        except Exception:
                            logger_mp.exception('Failed to toggle servo recording')
                    else:
                        logger_mp.warning('ServoJ toggle ignored: not currently recording')

                # get xr's tele data
                tele_data = tv_wrapper.get_tele_data()

                # ── Right A: 统一 press 跟踪 + 短按/长按检测 ──
                #  短按: release < 3s → arm toggle (relative_pose)
                #  长按: hold ≥ 3s    → mode switch
                _right_a_short = False
                if hasattr(tele_data, 'right_ctrl_aButton') and tele_data.right_ctrl_aButton:
                    _now = time.time()
                    if not getattr(tv_wrapper, '_prev_right_ctrl_aButton', False):
                        _right_a_press_start = _now
                    elif _now - _right_a_press_start >= 3.0 and not _right_a_long_press_triggered:
                        _right_a_long_press_triggered = True
                        if TELEOP_MODE == "relative_pose":
                            # Long press: switch back to relative_head
                            TELEOP_MODE = "relative_head"
                            LEFT_ARM_ACTIVE = False
                            RIGHT_ARM_ACTIVE = False
                            TELEOP_ACTIVE = False
                            tv_wrapper.reset_wrist_refs()
                            robot.reset_relative_pose_state('both')
                            tv_wrapper.reset_init_head_pose()
                            logger_mp.info("🔄 Right A long press: Switched to relative_head mode")
                    tv_wrapper._prev_right_ctrl_aButton = True
                else:
                    # Released
                    if getattr(tv_wrapper, '_prev_right_ctrl_aButton', False):
                        if not _right_a_long_press_triggered and _right_a_press_start > 0:
                            _right_a_short = True  # Released before 3s → short press
                    _right_a_press_start = 0.0
                    _right_a_long_press_triggered = False
                    tv_wrapper._prev_right_ctrl_aButton = False

                if TELEOP_MODE == "relative_head":
                    # ── relative_head: Left A toggles TELEOP_ACTIVE (existing logic) ──
                    if hasattr(tele_data, 'left_ctrl_aButton') and tele_data.left_ctrl_aButton:
                        if not getattr(tv_wrapper, '_prev_left_ctrl_aButton', False):
                            if TELEOP_ACTIVE:
                                TELEOP_ACTIVE = False
                                try:
                                    tv_wrapper.reset_init_head_pose()
                                except Exception:
                                    pass
                                logger_mp.info("XR Left A/X: Exit teleop -> TELEOP_ACTIVE=False (head ref cleared)")
                            else:
                                try:
                                    tv_wrapper.capture_init_head_pose()
                                except Exception:
                                    pass
                                TELEOP_ACTIVE = True
                                logger_mp.info("XR Left A/X: TELEOP_ACTIVE=True")
                        tv_wrapper._prev_left_ctrl_aButton = True
                    else:
                        tv_wrapper._prev_left_ctrl_aButton = False

                    # Right A in controller mode: quit teleoperate (existing)
                    if args.input_mode == "controller" and args.motion:
                        if tele_data.right_ctrl_aButton and not _right_a_long_press_triggered:
                            TELEOP_ACTIVE = False

                else:  # TELEOP_MODE == "relative_pose"
                    # ── relative_pose: Left A toggles LEFT_ARM_ACTIVE ──
                    if hasattr(tele_data, 'left_ctrl_aButton') and tele_data.left_ctrl_aButton:
                        if not getattr(tv_wrapper, '_prev_left_ctrl_aButton', False):
                            if LEFT_ARM_ACTIVE:
                                LEFT_ARM_ACTIVE = False
                                tv_wrapper.reset_left_wrist_ref()
                                logger_mp.info("XR Left A/X: Left hand DEACTIVATED")
                            else:
                                try:
                                    tv_wrapper.capture_left_wrist_ref()
                                except Exception:
                                    pass
                                LEFT_ARM_ACTIVE = True
                                robot.reset_relative_pose_state('right')
                                logger_mp.info("XR Left A/X: Left hand → Right arm ACTIVATED")
                        tv_wrapper._prev_left_ctrl_aButton = True
                    else:
                        tv_wrapper._prev_left_ctrl_aButton = False

                    # ── relative_pose: Right A short press (release < 3s) toggles RIGHT_ARM_ACTIVE ──
                    if _right_a_short:
                        if RIGHT_ARM_ACTIVE:
                            RIGHT_ARM_ACTIVE = False
                            tv_wrapper.reset_right_wrist_ref()
                            logger_mp.info("XR Right A: Right hand DEACTIVATED")
                        else:
                            try:
                                tv_wrapper.capture_right_wrist_ref()
                            except Exception:
                                pass
                            RIGHT_ARM_ACTIVE = True
                            robot.reset_relative_pose_state('left')
                            logger_mp.info("XR Right A: Right hand → Left arm ACTIVATED")

                    # ── Derive TELEOP_ACTIVE from arm activation ──
                    TELEOP_ACTIVE = LEFT_ARM_ACTIVE or RIGHT_ARM_ACTIVE

                # ── Pass teleop_mode to tele_data for RobotDriver ──
                tele_data.teleop_mode = TELEOP_MODE

                # ── XR B/Y buttons: 录制控制 ──────────────────────────
                # Left B/Y: 切换 ServoJ 连续轨迹录制
                if args.record and hasattr(tele_data, 'left_ctrl_bButton') and tele_data.left_ctrl_bButton:
                    if not getattr(tv_wrapper, '_prev_left_ctrl_bButton', False):
                        try:
                            if getattr(recorder_manager.recorder, '_servo_recording', False):
                                recorder_manager.stop_servo_recording()
                                if 'episode_writer' in globals() and episode_writer is not None:
                                    episode_writer.save_episode()
                                logger_mp.info("XR Left B/Y: ServoJ recording STOPPED")
                            else:
                                recorder_manager.ensure_program_active(
                                    program_name=args.task_name,
                                    description=args.task_desc,
                                    filepath=os.path.join(args.task_dir, args.task_name),
                                )
                                recorder_manager.start_servo_recording()
                                if 'episode_writer' in globals() and episode_writer is not None:
                                    episode_writer.create_episode()
                                logger_mp.info("XR Left B/Y: ServoJ recording STARTED")
                        except Exception:
                            logger_mp.exception('Failed to toggle ServoJ via XR')
                    tv_wrapper._prev_left_ctrl_bButton = True
                else:
                    tv_wrapper._prev_left_ctrl_bButton = False

                # Right B/Y: 记录单个 MoveJ 点
                if args.record and hasattr(tele_data, 'right_ctrl_bButton') and tele_data.right_ctrl_bButton:
                    if not getattr(tv_wrapper, '_prev_right_ctrl_bButton', False):
                        try:
                            recorder_manager.record_movej_point()
                            _pt_count = getattr(recorder_manager.recorder, 'total_points', 0)
                            _q = arm_ctrl.get_current_dual_arm_q()
                            _q_summary = '[' + ', '.join(f'{v:+.2f}' for v in _q[:3]) + ' ...]'
                            logger_mp.info(f"🎯 XR Right B/Y: MoveJ point #{_pt_count} recorded  |  joints[0:3]={_q_summary}")
                        except Exception:
                            logger_mp.exception('Failed to record MoveJ point via XR')
                    tv_wrapper._prev_right_ctrl_bButton = True
                else:
                    tv_wrapper._prev_right_ctrl_bButton = False

                # ── 底盘 & Torso 控制（独立于 TELEOP_ACTIVE，依赖 --motion）──
                _publish_base_cmd(tele_data)
                _publish_torso_cmd(tele_data)

                # ── 统一控制管线（RobotDriver）──
                control_cycle_timestamp_ns = time.time_ns()
                recording_snapshot = robot.step(tele_data)
                robot_step_done_timestamp_ns = time.time_ns()

                # ── 录制（机器人无关）──
                if args.record and getattr(recorder_manager.recorder, '_servo_recording', False):
                    try:
                        colors = {}
                        depths = {}
                        camera_timestamps_ns = {}
                        if camera_config['head_camera']['binocular']:
                            if head_img is not None:
                                colors[f"color_{0}"] = head_img.bgr[:, :camera_config['head_camera']['image_shape'][1]//2]
                                colors[f"color_{1}"] = head_img.bgr[:, camera_config['head_camera']['image_shape'][1]//2:]
                                if getattr(head_img, 'timestamp_ns', None) is not None:
                                    camera_timestamps_ns["color_0"] = head_img.timestamp_ns
                                    camera_timestamps_ns["color_1"] = head_img.timestamp_ns
                                if getattr(head_img, 'depth', None) is not None:
                                    _head_depth = head_img.depth
                                    depths[f"depth_{0}"] = _head_depth[:, :camera_config['head_camera']['image_shape'][1]//2]
                                    depths[f"depth_{1}"] = _head_depth[:, camera_config['head_camera']['image_shape'][1]//2:]
                        else:
                            if head_img is not None:
                                colors[f"color_{0}"] = head_img.bgr
                                if getattr(head_img, 'timestamp_ns', None) is not None:
                                    camera_timestamps_ns["color_0"] = head_img.timestamp_ns
                                if getattr(head_img, 'depth', None) is not None:
                                    depths[f"depth_{0}"] = head_img.depth
                        next_color_idx = len(colors)
                        if left_wrist_img is not None and getattr(left_wrist_img, 'bgr', None) is not None:
                            colors[f"color_{next_color_idx}"] = left_wrist_img.bgr
                            if getattr(left_wrist_img, 'timestamp_ns', None) is not None:
                                camera_timestamps_ns[f"color_{next_color_idx}"] = left_wrist_img.timestamp_ns
                            if getattr(left_wrist_img, 'depth', None) is not None:
                                depths[f"depth_{next_color_idx}"] = left_wrist_img.depth
                            next_color_idx += 1
                        if right_wrist_img is not None and getattr(right_wrist_img, 'bgr', None) is not None:
                            colors[f"color_{next_color_idx}"] = right_wrist_img.bgr
                            if getattr(right_wrist_img, 'timestamp_ns', None) is not None:
                                camera_timestamps_ns[f"color_{next_color_idx}"] = right_wrist_img.timestamp_ns
                            if getattr(right_wrist_img, 'depth', None) is not None:
                                depths[f"depth_{next_color_idx}"] = right_wrist_img.depth

                        # 构建 states/actions（与 RobotDriver._collect_recording_state 对齐）
                        left_arm_state = recording_snapshot.get("left_arm_state", [])
                        right_arm_state = recording_snapshot.get("right_arm_state", [])
                        left_arm_action = recording_snapshot.get("left_arm_action", [])
                        right_arm_action = recording_snapshot.get("right_arm_action", [])
                        body_state = recording_snapshot.get("body_state", [])
                        body_action = recording_snapshot.get("body_action", [])
                        ee_action = recording_snapshot.get("ee_action", [])

                        states = {
                            "left_arm": {"qpos": left_arm_state, "qvel": [], "torque": []},
                            "right_arm": {"qpos": right_arm_state, "qvel": [], "torque": []},
                            "left_ee": {"qpos": [], "qvel": [], "torque": []},
                            "right_ee": {"qpos": [], "qvel": [], "torque": []},
                            "body": {"qpos": body_state},
                        }
                        actions = {
                            "left_arm": {"qpos": left_arm_action, "qvel": [], "torque": []},
                            "right_arm": {"qpos": right_arm_action, "qvel": [], "torque": []},
                            "left_ee": {"qpos": [], "qvel": [], "torque": []},
                            "right_ee": {"qpos": [], "qvel": [], "torque": []},
                            "body": {"qpos": body_action},
                        }
                        camera_ts_values = list(camera_timestamps_ns.values())
                        camera_sync_skew_s = None
                        if camera_ts_values:
                            camera_sync_skew_s = (
                                max(camera_ts_values) - min(camera_ts_values)
                            ) / 1_000_000_000.0
                        alignment = {
                            "scheme": "lerobot_fps_grid",
                            "fps": args.frequency,
                            "tolerance_s": args.sync_tolerance_s,
                            "camera_sync_tolerance_s": args.camera_sync_tolerance_s,
                            "actual_timestamps_ns": {
                                "control_cycle": control_cycle_timestamp_ns,
                                "robot_step_done": robot_step_done_timestamp_ns,
                                "cameras": camera_timestamps_ns,
                            },
                            "camera_sync_skew_s": camera_sync_skew_s,
                            "camera_sync_within_tolerance": (
                                None if camera_sync_skew_s is None
                                else camera_sync_skew_s <= args.camera_sync_tolerance_s
                            ),
                        }
                        # TODO: 回写 sim_state

                        if 'episode_writer' in globals() and episode_writer is not None:
                            episode_writer.add_item(
                                colors=colors,
                                depths=depths,
                                states=states,
                                actions=actions,
                                alignment=alignment,
                            )
                    except Exception:
                        logger_mp.exception("Failed to record trajectory data")
                        raise

                current_time = time.time()
                time_elapsed = current_time - start_time
                sleep_time = max(0, (1 / args.frequency) - time_elapsed)
                time.sleep(sleep_time)
                logger_mp.debug(f"main process sleep: {sleep_time}")

        # ======== 退出遥操作后的清理 ========
        # 底盘归零
        if MOTION_ACTIVE:
            try:
                if hasattr(arm_ctrl, '_ros_node') and hasattr(arm_ctrl._ros_node, 'publish_base_cmd'):
                    arm_ctrl._ros_node.publish_base_cmd(0.0, 0.0, 0.0)
                    logger_mp.info("Base velocity reset to zero.")
            except Exception:
                pass

        # 自动停止可能仍在进行的 ServoJ 录制
        if args.record:
            try:
                if getattr(recorder_manager.recorder, '_servo_recording', False):
                    recorder_manager.stop_servo_recording()
                    logger_mp.info("ServoJ recording stopped automatically (teleop exited).")
            except Exception:
                logger_mp.exception("Failed to auto-stop ServoJ recording")
            RECORD_RUNNING = False
            if args.sim:
                pass  # sim reset TODO

    except KeyboardInterrupt:
        logger_mp.info("⛔ KeyboardInterrupt, exiting program...")
    except Exception:
        import traceback
        logger_mp.error(traceback.format_exc())
    finally:
        # 1. restore terminal (sshkeyboard leaves it in raw mode)
        try:
            stop_listening()
        except Exception:
            pass

        # 2. send robot home via ArmRequest
        try:
            if 'arm_ctrl' in locals():
                arm_ctrl.go_home()
        except Exception as e:
            logger_mp.error(f"Failed to go_home: {e}")

        # auto-save any active recording
        try:
            if args.record and 'recorder_manager' in locals():
                if getattr(recorder_manager.recorder, '_servo_recording', False):
                    recorder_manager.stop_servo_recording()
                if getattr(recorder_manager.recorder, '_program_active', False):
                    recorder_manager.stop_program()
                    logger_mp.info("Recording saved on exit.")
        except Exception as e:
            logger_mp.error(f"Failed to save recording on exit: {e}")

        # close episode writer (flushes any pending episode and stops its worker thread)
        try:
            if args.record and 'episode_writer' in locals() and episode_writer is not None:
                episode_writer.close()
                logger_mp.info("Episode writer closed on exit.")
        except Exception as e:
            logger_mp.error(f"Failed to close episode writer on exit: {e}")

        # 3. graceful rclpy shutdown (prevents "terminate called without an active exception")
        try:
            if 'arm_ctrl' in locals() and hasattr(arm_ctrl, 'stop'):
                arm_ctrl.stop()
        except Exception as e:
            logger_mp.error(f"Failed to stop arm_ctrl: {e}")
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass

        # 4. stop Vuer subprocess + clean up shared memory
        try:
            if 'tv_wrapper' in locals() and hasattr(tv_wrapper, 'tvuer') and tv_wrapper.tvuer is not None:
                tv_wrapper.tvuer.close()
                logger_mp.info("Vuer subprocess terminated and shared memory cleaned.")
        except Exception as e:
            logger_mp.error(f"Failed to stop Vuer: {e}")

        # 5. force exit if rclpy.shutdown() hangs (ROS 2 DDS known issue)
        threading.Thread(
            target=lambda: (time.sleep(5.0), os._exit(0)),
            daemon=True,
        ).start()
'''
conda deactivate
source /opt/ros/humble/setup.sh
source /media/ai/d9787eb9-5947-4134-be08-d9b5ed71bdde/tele_robot/complete_test/topstar_ros2_redeploy_20260518/setup.sh
conda activate tv
cd /media/ai/d9787eb9-5947-4134-be08-d9b5ed71bdde/tele_robot/complete_test/tele_robot/teleop
python teleop_hand_and_arm.py  --frequency 10 --arm TOPSTAR_H1 --img-server-ip 192.168.0.2 --input-mode controller --arm-scale 1.5 --replay --replay-file "/media/ai/d9787eb9-5947-4134-be08-d9b5ed71bdde/tele_robot/complete_test/tele_robot/teleop/utils/data/pick cube/pick cube.json"
python teleop_hand_and_arm.py  --frequency 20 --input-mode controller --robot TOPSTAR_H1 --control-mode arms_head --ee suction_cup --img-server-ip 192.168.0.2 --arm-scale 1.5 --record
python teleop_hand_and_arm.py  --frequency 100 --input-mode controller --robot TOPSTAR_H2 --control-mode arms_only --arm-scale 1.0 --record

'''
