import json
import hmac
import hashlib
import time
import re
from datetime import datetime
from typing import List, Optional
from pathlib import Path
import os
import threading
import math
import logging

import numpy as np

logger = logging.getLogger(__name__)


def _resolve_fk_func(ik):
    """Return a callable f(q_deg) -> (left_xyz, right_xyz) for the given IK instance.

    Supports:
    - IK classes with explicit ``get_dual_arm_ee_positions(q_deg)`` (TOPSTAR_H1)
    - Generic pinocchio-based IK classes (G1_29, G1_23, H1_2, H1, TH010) that have
      ``reduced_robot``, ``L_hand_id``, ``R_hand_id``.
    Returns None if no FK capability is available.
    """
    if ik is None:
        return None

    # Preferred: explicit method (TOPSTAR_H1_ArmIK)
    if hasattr(ik, 'get_dual_arm_ee_positions'):
        return ik.get_dual_arm_ee_positions

    # Fallback: generic pinocchio-based IK
    if hasattr(ik, 'reduced_robot') and hasattr(ik, 'L_hand_id') and hasattr(ik, 'R_hand_id'):
        # Lazy-import pinocchio inside the closure so record.py stays lightweight
        def _generic_fk(q_deg):
            import pinocchio as _pin
            q_rad = np.deg2rad(np.asarray(q_deg, dtype=float))
            nq = ik.reduced_robot.model.nq
            q_full = np.zeros(nq)
            n = min(len(q_rad), nq)
            q_full[:n] = q_rad[:n]
            _pin.forwardKinematics(ik.reduced_robot.model, ik.reduced_robot.data, q_full)
            _pin.updateFramePlacements(ik.reduced_robot.model, ik.reduced_robot.data)
            left_pos = ik.reduced_robot.data.oMf[ik.L_hand_id].translation.copy()
            right_pos = ik.reduced_robot.data.oMf[ik.R_hand_id].translation.copy()
            return left_pos, right_pos
        return _generic_fk

    return None


class Recorder:
    def __init__(self, secret_key: bytes = None, buffer_size: int = 1000):
        """
        :param secret_key: 签名密钥
        :param buffer_size: 缓冲区大小
        """
        self.secret_key = secret_key or b"secret_key_2026"
        self.buffer_size = buffer_size
        self.point_buffer = []
        self.metadata = {
            "created_at": None,
            "modified_at": None,
            "program_name": None,
            "description": "",
            "total_points": 0,
            "frequency": 0,
        }
        self.current_file = None
        self._points_temp_file = None
        self.json_file = None
        self.total_points = 0
        self._first_point = True
        self._program_active = False

        self._servo_recording = False
        self._servo_buffer = []
        self._servo_joint_count = None

    def start_program(self, program_name: str, description: str = "", filepath: Optional[str] = None, frequency: int = 500,
                      ee: Optional[str] = None,
                      control_mode: Optional[str] = None,
                      robot_model: Optional[str] = None):
        if self.json_file and not self.json_file.closed:
            self.json_file.close()
        self.json_file = None

        if self._points_temp_file and self._points_temp_file.exists():
            self._points_temp_file.unlink()

        self.metadata["program_name"] = program_name
        self.metadata["description"] = description
        self.metadata["created_at"] = datetime.now().isoformat()
        self.metadata["frequency"] = frequency
        self.metadata.pop("modified_at", None)
        if ee is not None:
            self.metadata["ee"] = ee
        else:
            self.metadata.pop("ee", None)
        if control_mode is not None:
            self.metadata["control_mode"] = control_mode
        else:
            self.metadata.pop("control_mode", None)
        if robot_model is not None:
            self.metadata["robot_model"] = robot_model
        else:
            self.metadata.pop("robot_model", None)
        self.total_points = 0
        self.point_buffer = []
        self._first_point = True
        self._program_active = False
        self._servo_recording = False
        self._servo_buffer = []
        self._servo_joint_count = None
        self.current_file = None
        self._points_temp_file = None

        if not filepath:
            raise ValueError("filepath 不能为空，请传入输出目录")

        output_dir = Path(filepath)
        output_dir.mkdir(parents=True, exist_ok=True)

        safe_name = re.sub(r'[\\/:*?"<>|]+', '_', program_name.strip())
        if not safe_name:
            safe_name = "program"

        self.current_file = output_dir / f"{safe_name}.json"
        self._points_temp_file = output_dir / f"{safe_name}.points.tmp"

        try:
            self.json_file = open(self._points_temp_file, 'w', encoding='utf-8')
        except IOError as exc:
            raise RuntimeError(f"无法创建临时文件: {exc}")

        self.json_file = open(self._points_temp_file, 'w', encoding='utf-8')
        self._first_point = True
        self._program_active = True

        print(f"✅ 轨迹记录已初始化: {program_name}")

    def is_ready(self) -> bool:
        return not self._program_active

    def _ensure_program_active(self):
        if not self._program_active or not self.json_file or self.json_file.closed:
            raise RuntimeError("请先调用 start_program 开始程序，finish 后如需继续请重新开始新的 program")

    def _add_point(self, motion_type: str, comment: Optional[str] = "", **kwargs):
        self._ensure_program_active()
        point = {
            "id": self.total_points,
            "type": motion_type,
            **kwargs,
        }
        self.point_buffer.append(point)
        self.total_points += 1

        if len(self.point_buffer) >= self.buffer_size:
            self._flush_buffer()

    def add_MoveJ(self,
                  joint_pose: List[float],
                  velocity: int = 0,
                  acceleration: int = 0,
                  cnt: int = -1,
                  comment: Optional[str] = "",
                  ee_action: Optional[List[float]] = None,
                  **extra_fields):
        extra = dict(extra_fields)
        if ee_action is not None:
            extra["ee_action"] = ee_action
        self._add_point(
            motion_type="MoveJ",
            joint_pose=joint_pose,
            **extra,
        )

    def add_MoveL(self,
                  cartesian_pose: List[float],
                  velocity: int = 0,
                  acceleration: int = 0,
                  user_frame: int = 0,
                  tool_frame: int = 0,
                  cfg: Optional[List[int]] = None,
                  cnt: int = -1,
                  comment: Optional[str] = "",
                  **extra_fields):
        cfg_value = [0, 0, 0, 0] if cfg is None else cfg
        self._add_point(
            motion_type="MoveL",
            cartesian_pose=cartesian_pose,
            user_frame=user_frame,
            tool_frame=tool_frame,
            cfg=cfg_value,
            **extra_fields,
        )

    def add_SetDO(self, io: int, value: bool, delay_s: float = 0.0, **extra_fields):
        self._ensure_program_active()
        point = {
            "id": self.total_points,
            "type": "SetDO",
            "io": io,
            "value": value,
            "delay_s": delay_s,
            **extra_fields,
        }
        self.point_buffer.append(point)
        self.total_points += 1

        if len(self.point_buffer) >= self.buffer_size:
            self._flush_buffer()

    def start_servo_recording(self):
        self._ensure_program_active()

        if self._servo_recording:
            raise RuntimeError("ServoJ 录制已在进行中，请先调用 stop_servo_recording 停止")

        self._servo_recording = True
        self._servo_buffer = []
        self._servo_joint_count = None
        print("✅ ServoJ 轨迹记录已开始")

    def add_ServoJ(self,
                   joint_pose: List[float],
                   comment: Optional[str] = "",
                   ee_action: Optional[List[float]] = None,
                   **extra_fields):
        self._ensure_program_active()

        if not self._servo_recording:
            raise RuntimeError("请先调用 start_servo_recording 开始 ServoJ 记录")

        if not joint_pose:
            raise ValueError("joint_pose 不能为空")

        current_joint_count = len(joint_pose)
        if self._servo_joint_count is None:
            self._servo_joint_count = current_joint_count
        elif current_joint_count != self._servo_joint_count:
            raise ValueError(
                f"ServoJ joint_pose 维度不一致：期望 {self._servo_joint_count}，实际 {current_joint_count}"
            )

        point = {
            "id": self.total_points,
            "type": "ServoJ",
            "joint_pose": list(joint_pose),
            **extra_fields,
        }
        if ee_action is not None:
            point["ee_action"] = ee_action

        self._servo_buffer.append(point)
        self.total_points += 1

    def stop_servo_recording(self, smooth_window_size: int = 5, smooth_passes: int = 1):
        if not self._servo_recording:
            raise RuntimeError("ServoJ 录制未进行中")

        self._servo_recording = False

        if not self._servo_buffer:
            print("⚠️ ServoJ 缓冲为空，无需平滑")
            return

        self._smooth_servo_segment(smooth_window_size, smooth_passes)

        for point in self._servo_buffer:
            self.point_buffer.append(point)
            if len(self.point_buffer) >= self.buffer_size:
                self._flush_buffer()

        self._servo_buffer = []
        self._servo_joint_count = None

        print("✅ ServoJ 轨迹记录已停止并平滑")

    def _smooth_servo_segment(self, window_size: int = 5, passes: int = 1):
        if window_size < 1 or window_size % 2 == 0:
            raise ValueError("window_size 必须是正奇数")
        if passes < 1:
            raise ValueError("passes 必须大于 0")

        if not self._servo_buffer:
            return

        for _ in range(passes):
            source_poses = [list(point["joint_pose"]) for point in self._servo_buffer]
            joint_count = len(source_poses[0])
            half_window = window_size // 2

            for pose in source_poses:
                if len(pose) != joint_count:
                    raise ValueError("所有 ServoJ 点的 joint_pose 长度必须一致")

            smoothed_poses = []
            for center in range(len(source_poses)):
                start = max(0, center - half_window)
                end = min(len(source_poses), center + half_window + 1)
                window = source_poses[start:end]
                smoothed_pose = [
                    sum(sample[joint_index] for sample in window) / len(window)
                    for joint_index in range(joint_count)
                ]
                smoothed_poses.append(smoothed_pose)

            for point_idx, smoothed_pose in enumerate(smoothed_poses):
                self._servo_buffer[point_idx]['joint_pose'] = smoothed_pose

    def add_precise_point(self,
                          joint_pose: List[float],
                          velocity: int = 10,
                          acceleration: int = 10,
                          cnt: int = -1,
                          comment: str = "精确定位点",
                          **extra_fields):
        self._ensure_program_active()

        if self._servo_recording:
            print("⚠️ ServoJ 录制进行中，先停止录制")
            self.stop_servo_recording()

        self._add_point(
            motion_type="MoveJ",
            joint_pose=joint_pose,
            **extra_fields,
        )

    def _flush_buffer(self):
        if not self.point_buffer:
            return

        self._ensure_program_active()

        for point in self.point_buffer:
            if not self._first_point:
                self.json_file.write(',\n')
            self._first_point = False
            json.dump(point, self.json_file, ensure_ascii=False)
        self.json_file.flush()
        self.point_buffer = []

    def finish(self, fk_func=None):
        """Finalize recording: flush buffer, optionally compute MoveJ durations via FK, write output.

        Args:
            fk_func: Optional callable f(q_deg) -> (left_xyz, right_xyz) for MoveJ duration calc.
        """
        if not self.json_file:
            return

        finalize_error = None

        try:
            if self._servo_recording:
                print("⚠️ ServoJ 录制进行中，自动停止并平滑")
                self.stop_servo_recording()

            self._flush_buffer()
        finally:
            if self.json_file and not self.json_file.closed:
                self.json_file.close()
            self.json_file = None

        try:
            self._finalize_output_file(fk_func=fk_func)
            self._generate_signature()
        except Exception as exc:
            finalize_error = exc
        finally:
            self._program_active = False

        if finalize_error is not None:
            raise RuntimeError(f"finish 失败: {finalize_error}") from finalize_error

        print(f"✅ 遥操程序已完成，共 {self.total_points} 个轨迹点")
        print(f"📁 文件大小: {os.path.getsize(self.current_file) / 1024 / 1024:.2f} MB")

    def close(self):
        if self._program_active:
            self.finish()

    def _finalize_output_file(self, fk_func=None):
        metadata = {
            **self.metadata,
            "total_points": self.total_points,
            "modified_at": datetime.now().isoformat(),
        }

        # ── Read all points from temp file ──
        points = []
        if self._points_temp_file and self._points_temp_file.exists():
            with open(self._points_temp_file, 'r', encoding='utf-8') as temp:
                content = temp.read()
            # temp file contains JSON objects separated by ",\n"
            # Wrap in array brackets for safe parsing
            try:
                points = json.loads("[" + content + "]")
            except json.JSONDecodeError:
                # fallback: try parsing line by line
                points = []
                for line in content.split("\n"):
                    line = line.strip().rstrip(",")
                    if line:
                        try:
                            points.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass

        # ── Compute MoveJ durations via FK ──
        if fk_func is not None and len(points) > 0:
            points = self._compute_movej_durations(points, fk_func)

        with open(self.current_file, 'w', encoding='utf-8') as out:
            metadata_json = json.dumps(metadata, indent=2, ensure_ascii=False)
            out.write('{\n  "metadata": ' + metadata_json + ',\n  "points": [\n')

            for i, pt in enumerate(points):
                if i > 0:
                    out.write(',\n')
                json.dump(pt, out, ensure_ascii=False)

            out.write('\n  ]\n}\n')

        if self._points_temp_file and self._points_temp_file.exists():
            self._points_temp_file.unlink()

    @staticmethod
    def _compute_movej_durations(points: list, fk_func, speed_ms: float = 0.01):
        """Compute duration for each MoveJ point based on FK straight-line distance.

        For each MoveJ point, compute the EE positions of this point and the previous
        point (of any type).  Left and right arms are evaluated independently and the
        **larger** of the two straight-line distances is used:

            duration = max(d_left, d_right) / speed_ms

        Args:
            points: list of point dicts
            fk_func: callable f(q_deg) -> (left_xyz, right_xyz), each shape (3,)
            speed_ms: desired end-effector speed in m/s (default 0.1)

        Returns:
            Modified list of points with 'duration' field added to MoveJ points.
        """
        _prev_ee = None  # (left_xyz, right_xyz) of the last point

        for pt in points:
            if pt.get("type") != "MoveJ":
                # For non-MoveJ points, still track EE position for next MoveJ
                jp = pt.get("joint_pose")
                if jp and len(jp) >= 2:
                    try:
                        _prev_ee = fk_func(jp)
                    except Exception:
                        _prev_ee = None
                continue

            jp = pt.get("joint_pose")
            if not jp or len(jp) < 2:
                continue

            try:
                cur_ee = fk_func(jp)
            except Exception:
                cur_ee = None

            if _prev_ee is not None and cur_ee is not None:
                dl = float(np.linalg.norm(cur_ee[0] - _prev_ee[0]))
                dr = float(np.linalg.norm(cur_ee[1] - _prev_ee[1]))
                dist = max(dl, dr)
                pt["duration"] = round(dist / speed_ms, 3)

            _prev_ee = cur_ee

        return points

    def _generate_signature(self):
        with open(self.current_file, 'rb') as f:
            file_content = f.read()

        signature = hmac.new(
            self.secret_key,
            file_content,
            hashlib.sha256,
        ).hexdigest()

        sig_path = self.current_file.with_suffix('.sig')
        with open(sig_path, 'w') as f:
            f.write(signature)

        print(f"✅ 签名已生成: {sig_path}")


class RecorderManager:
    def __init__(self, recorder, arm_ctrl, ik=None, frequency: float = 200.0,
                 ee_type: Optional[str] = None,
                 ee_action_getter=None,   # Callable[[], Optional[List[float]]]
                 control_mode=None,       # ControlMode — 按模式采样
                 robot_model=None):       # str — 写入轨迹 metadata
        self.recorder = recorder
        self.arm_ctrl = arm_ctrl
        self.ik = ik
        self.frequency = float(frequency) if frequency and frequency > 0 else 200.0
        self.ee_type = ee_type
        self.ee_action_getter = ee_action_getter
        self.control_mode = control_mode
        self.robot_model = robot_model

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._stop_event = threading.Event()

    def start(self):
        self._stop_event.clear()
        if not self._thread.is_alive():
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def stop(self, timeout: float = 1.0):
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=timeout)

    def start_program(self, program_name: str, description: str, filepath: str, frequency: int):
        self.recorder.start_program(
            program_name, description, filepath, frequency,
            ee=self.ee_type,
            control_mode=self.control_mode.value if self.control_mode else None,
            robot_model=self.robot_model,
        )

    def stop_program(self):
        try:
            fk_func = _resolve_fk_func(self.ik)
            self.recorder.finish(fk_func=fk_func)
        except Exception:
            logger.exception("RecorderManager: finish failed")

    def start_servo_recording(self):
        self.recorder.start_servo_recording()

    def stop_servo_recording(self, smooth_window_size: int = 5, smooth_passes: int = 1):
        self.recorder.stop_servo_recording(smooth_window_size, smooth_passes)

    def add_precise_point(self, joint_pose, **kwargs):
        jp_deg = self._to_degrees(list(joint_pose))
        self.recorder.add_precise_point(joint_pose=jp_deg, **kwargs)

    def _to_degrees(self, joint_pose):
        try:
            return [float(x) * 180.0 / math.pi for x in joint_pose]
        except Exception:
            return [float(x) for x in joint_pose]

    def ensure_program_active(self, program_name: str = "record", description: str = "", filepath: str = "./utils/data/"):
        """Auto-start a recording program if not already active."""
        if getattr(self.recorder, '_program_active', False):
            return
        self.start_program(program_name, description, filepath, int(self.frequency))
        logger.info(f"Recording program auto-started: {program_name}")

    def record_movej_point(self):
        """Snapshot current joint angles and record as a MoveJ point."""
        self.ensure_program_active()
        if hasattr(self.arm_ctrl, 'get_current_dual_arm_q'):
            q = self.arm_ctrl.get_current_dual_arm_q()
        elif hasattr(self.arm_ctrl, 'get_current_arm_q'):
            q = self.arm_ctrl.get_current_arm_q()
        else:
            raise RuntimeError('RecorderManager: arm_ctrl does not provide a supported state getter')
        if q is None or len(q) == 0:
            raise RuntimeError('RecorderManager: arm state is empty')
        joint_pose_deg = self._to_degrees([float(x) for x in q])
        ee_action = None
        if self.ee_action_getter is not None:
            try:
                ee_action = self.ee_action_getter()
            except Exception:
                logger.exception('RecorderManager: failed to sample EE action in record_movej_point')
        self.recorder.add_MoveJ(joint_pose_deg, ee_action=ee_action)

    def _run(self):
        period = 1.0 / max(1.0, self.frequency)
        next_time = time.time()
        while not self._stop_event.is_set():
            now = time.time()
            if now < next_time:
                time.sleep(next_time - now)
                continue
            next_time += period

            if not getattr(self.recorder, '_program_active', False):
                continue

            # 按 control_mode 采样
            mode = self.control_mode
            if mode is not None and hasattr(self.arm_ctrl, 'get_current_joint_state'):
                q = self.arm_ctrl.get_current_joint_state(mode)
            elif hasattr(self.arm_ctrl, 'get_current_dual_arm_q'):
                q = self.arm_ctrl.get_current_dual_arm_q()
            elif hasattr(self.arm_ctrl, 'get_current_arm_q'):
                q = self.arm_ctrl.get_current_arm_q()
            else:
                logger.exception('RecorderManager: arm_ctrl does not provide a supported state getter')
                raise RuntimeError('RecorderManager: missing state getter')

            if q is None:
                logger.exception('RecorderManager: arm state is None')
                raise RuntimeError('RecorderManager: arm state is None')

            try:
                joint_pose = [float(x) for x in q]
            except Exception:
                logger.exception('RecorderManager: failed to convert arm state to list')
                raise

            if len(joint_pose) == 0:
                logger.exception('RecorderManager: arm state is empty')
                raise RuntimeError('RecorderManager: arm state is empty')

            joint_pose_deg = self._to_degrees(joint_pose)

            # 采样头部（ARMS_HEAD 时单独记录）
            head_pose_deg = None
            if mode is not None and mode.solve_head:
                try:
                    head_q = self.arm_ctrl.get_head_q()
                    if head_q is not None and len(head_q) >= 2:
                        head_pose_deg = self._to_degrees(
                            [float(head_q[0]), float(head_q[1])]
                        )
                except Exception:
                    pass

            # Sample EE action if getter is available
            ee_action = None
            if self.ee_action_getter is not None:
                try:
                    ee_action = self.ee_action_getter()
                except Exception:
                    logger.exception('RecorderManager: failed to sample EE action')

            try:
                if getattr(self.recorder, '_servo_recording', False):
                    extra = {}
                    if head_pose_deg is not None:
                        extra["head_pose"] = head_pose_deg
                    self.recorder.add_ServoJ(joint_pose_deg, ee_action=ee_action,
                                             **extra)
                # MoveJ points are recorded manually via record_movej_point()
            except Exception:
                logger.exception('RecorderManager: failed to add point')
                raise
