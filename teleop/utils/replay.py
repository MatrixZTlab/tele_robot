from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import hmac
import json
import logging
import queue
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class ReplayState(str, Enum):
    IDLE = "Idle"
    LOADED = "Loaded"
    READY = "Ready"
    PLAYING = "Playing"
    PAUSED = "Paused"
    STOPPED = "Stopped"
    FINISHED = "Finished"
    ERROR = "Error"
    CLOSED = "Closed"


@dataclass
class ReplayMetadataSummary:
    program_name: Optional[str] = None
    description: Optional[str] = None
    total_points: int = 0
    created_at: Optional[str] = None
    modified_at: Optional[str] = None
    frequency: int = 0


@dataclass
class ReplayAnalysisReport:
    total_points: int
    point_type_stats: Dict[str, int] = field(default_factory=dict)
    servo_segment_count: int = 0
    joint_dim_consistency: bool = True
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


@dataclass
class ReplayOptions:
    speed_scale: float = 1.0
    dry_run: bool = False


@dataclass
class ReplayStatus:
    session_id: str
    state: ReplayState
    producer_index: int
    consumed_index: int
    progress: float
    eta_seconds: Optional[float] = None
    last_error: Optional[str] = None


@dataclass
class ReplayEvent:
    event_type: str
    session_id: str
    timestamp: float
    payload: Dict[str, Any] = field(default_factory=dict)


ReplayEventCallback = Callable[[ReplayEvent], None]
ReplayEventTypes = Optional[Set[str]]


@dataclass
class _ReplayEventSubscription:
    callback: ReplayEventCallback
    event_types: ReplayEventTypes = None


@dataclass
class ReplayPointBufferEntry:
    point: Dict[str, Any]
    session_id: str = ""


@dataclass
class _ReplayRecord:
    path: str
    metadata: Dict[str, Any]
    points: List[Dict[str, Any]]
    state: ReplayState = ReplayState.IDLE


@dataclass
class _ReplaySession:
    session_id: str
    trajectory_id: str
    state: ReplayState
    options: ReplayOptions
    start_index: int
    end_index: int
    producer_index: int
    consumed_index: int = 0
    running_event: threading.Event = field(default_factory=threading.Event)
    stop_event: threading.Event = field(default_factory=threading.Event)
    _exec_lock: threading.Lock = field(default_factory=threading.Lock)
    thread: Optional[threading.Thread] = None
    last_error: Optional[str] = None
    subscription_callbacks: Dict[str, _ReplayEventSubscription] = field(default_factory=dict)
    started_at: Optional[float] = None
    ended_at: Optional[float] = None


class ReplayConsumer:
    def __init__(self, replay_queue: queue.Queue, dispatcher: Callable[[Dict[str, Any]], None], frequency: Optional[float] = None):
        self.replay_queue = replay_queue
        self.dispatcher = dispatcher
        self.frequency = float(frequency) if frequency else None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._stop_event = threading.Event()

    def start(self) -> None:
        self._stop_event.clear()
        if not self._thread.is_alive():
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=timeout)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                item = self.replay_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                point = item.get("point") if isinstance(item, dict) else item
                try:
                    self.dispatcher(point)
                except Exception:
                    logger.exception("ReplayConsumer dispatcher failed for point id=%s", point.get("id"))
            finally:
                try:
                    self.replay_queue.task_done()
                except Exception:
                    pass

            if self.frequency and self.frequency > 0:
                time.sleep(1.0 / self.frequency)


_replay_queue: queue.Queue = queue.Queue(maxsize=2000)


def get_replay_queue() -> queue.Queue:
    return _replay_queue


def replay_executor(point: Dict[str, Any], options, session_id: str = "") -> None:
    # 阻塞写入，确保不丢点；生产速度受消费者消费速度节制
    _replay_queue.put({"point": point, "session_id": session_id, "options": options})


def clear_replay_queue() -> None:
    try:
        while True:
            _replay_queue.get_nowait()
    except queue.Empty:
        return


class Replay:
    def __init__(self, buffer_size: int = 1000, replay_executor: Optional[Callable[[Dict[str, Any], ReplayOptions, str], None]] = None):
        self._lock = threading.RLock()
        self._replays: Dict[str, _ReplayRecord] = {}
        self._sessions: Dict[str, _ReplaySession] = {}
        self._buffer_size = buffer_size
        self._buffer_lock = threading.RLock()
        self._point_buffer: Deque[ReplayPointBufferEntry] = deque(maxlen=buffer_size)
        self._external_replay_executor = replay_executor

    def set_replay_executor(self, replay_executor: Optional[Callable[[Dict[str, Any], ReplayOptions, str], None]]) -> None:
        self._external_replay_executor = replay_executor

    def _replay_executor(self, point: Dict[str, Any], options: ReplayOptions, session_id: str = "") -> None:
        if options.dry_run:
            return

        if self._external_replay_executor is not None:
            self._external_replay_executor(point, options, session_id)
            return

        entry = ReplayPointBufferEntry(point=deepcopy(point), session_id=session_id)
        with self._buffer_lock:
            self._point_buffer.append(entry)

    def get_point_entry(self) -> Optional[ReplayPointBufferEntry]:
        with self._buffer_lock:
            if not self._point_buffer:
                return None
            entry = self._point_buffer.popleft()
        if entry.session_id:
            with self._lock:
                session = self._sessions.get(entry.session_id)
                if session is not None:
                    session.consumed_index += 1
        return entry

    def clear_point_buffer(self) -> None:
        with self._buffer_lock:
            self._point_buffer.clear()

    def _get_replay(self, replay_id: str) -> _ReplayRecord:
        replay = self._replays.get(replay_id)
        if replay is None:
            raise KeyError(f"replay_id 不存在: {replay_id}")
        return replay

    def _get_session(self, session_id: str) -> _ReplaySession:
        session = self._sessions.get(session_id)
        if session is None:
            raise KeyError(f"session_id 不存在: {session_id}")
        return session

    def _emit_event(self, session: _ReplaySession, event_type: str, payload: Optional[Dict[str, Any]] = None) -> None:
        event = ReplayEvent(
            event_type=event_type,
            session_id=session.session_id,
            timestamp=time.time(),
            payload=payload or {},
        )
        for subscription in list(session.subscription_callbacks.values()):
            if subscription.event_types is not None and event_type not in subscription.event_types:
                continue
            try:
                subscription.callback(event)
            except Exception:
                pass

    def _set_state(self, session: _ReplaySession, new_state: ReplayState) -> None:
        old_state = session.state
        session.state = new_state
        self._emit_event(
            session,
            "state_changed",
            {
                "from": old_state.value,
                "to": new_state.value,
                "producer_index": session.producer_index,
            },
        )

    def _validate_signature(self, path: Path, secret_key: bytes) -> None:
        sig_path = path.with_suffix(".sig")
        if not sig_path.exists():
            raise RuntimeError(f"签名文件不存在: {sig_path}")

        with open(path, "rb") as file_obj:
            file_content = file_obj.read()
        expected = hmac.new(secret_key, file_content, hashlib.sha256).hexdigest()

        with open(sig_path, "r", encoding="utf-8") as sig_obj:
            actual = sig_obj.read().strip()

        if not hmac.compare_digest(expected, actual):
            raise RuntimeError("签名校验失败")

    def _normalize_options(self, options: Optional[ReplayOptions]) -> ReplayOptions:
        normalized = options or ReplayOptions()
        if normalized.speed_scale <= 0:
            raise ValueError("speed_scale 必须大于 0")
        return normalized

    def _session_progress(self, session: _ReplaySession) -> float:
        total = max(1, session.end_index - session.start_index + 1)
        done = max(0, min(session.consumed_index - session.start_index, total))
        return done / total

    def _execute_single_point(self, session: _ReplaySession, point: Dict[str, Any]) -> None:
        self._emit_event(session, "point_started", {"index": session.producer_index, "type": point.get("type")})
        self._replay_executor(point, session.options, session.session_id)
        self._emit_event(session, "point_finished", {"index": session.producer_index, "type": point.get("type")})

    def _replay_loop(self, session_id: str) -> None:
        session = self._get_session(session_id)
        replay = self._get_replay(session.trajectory_id)

        try:
            while not session.stop_event.is_set():
                if session.producer_index > session.end_index:
                    if session.consumed_index > session.end_index:
                        session.ended_at = time.time()
                        self._set_state(session, ReplayState.FINISHED)
                        self._emit_event(session, "replay_finished", {"current_index": session.producer_index})
                        return
                    time.sleep(0.01)
                    continue

                if not session.running_event.is_set():
                    time.sleep(0.01)
                    continue

                with session._exec_lock:
                    point = replay.points[session.producer_index]
                    self._execute_single_point(session, point)
                    session.producer_index += 1

        except Exception as exc:
            session.last_error = str(exc)
            session.ended_at = time.time()
            self._set_state(session, ReplayState.ERROR)
            self._emit_event(session, "error", {"message": str(exc), "current_index": session.producer_index})

    def load_replay(self, path: str, secret_key: Optional[bytes] = None, verify_signature: bool = True) -> tuple[str, ReplayMetadataSummary]:
        replay_path = Path(path)
        if not replay_path.exists() or not replay_path.is_file():
            raise FileNotFoundError(f"轨迹文件不存在: {path}")

        if verify_signature:
            if secret_key is None:
                raise ValueError("verify_signature=True 时必须提供 secret_key")
            self._validate_signature(replay_path, secret_key)

        with open(replay_path, "r", encoding="utf-8") as file_obj:
            data = json.load(file_obj)

        if not isinstance(data, dict):
            raise ValueError("轨迹文件格式错误：顶层应为对象")

        metadata = data.get("metadata", {})
        points = data.get("points", [])
        if not isinstance(metadata, dict):
            raise ValueError("轨迹文件格式错误：metadata 应为对象")
        if not isinstance(points, list):
            raise ValueError("轨迹文件格式错误：points 应为数组")

        replay_id = str(uuid.uuid4())
        record = _ReplayRecord(path=str(replay_path), metadata=metadata, points=points)
        record.state = ReplayState.LOADED
        with self._lock:
            self._replays[replay_id] = record

        summary = ReplayMetadataSummary(
            program_name=metadata.get("program_name"),
            description=metadata.get("description"),
            total_points=int(metadata.get("total_points", len(points)) or 0),
            created_at=metadata.get("created_at"),
            modified_at=metadata.get("modified_at"),
            frequency=int(metadata.get("frequency", 0) or 0),
        )
        return replay_id, summary

    def analyze_replay(self, replay_id: str) -> ReplayAnalysisReport:
        replay = self._get_replay(replay_id)
        points = replay.points

        type_stats: Dict[str, int] = {}
        warnings: List[str] = []
        errors: List[str] = []

        servo_segment_count = 0
        in_servo_segment = False
        joint_dim_consistency = True
        current_segment_joint_dim: Optional[int] = None

        supported_types = {"MoveJ", "MoveL", "ServoJ", "SetDO"}

        for idx, point in enumerate(points):
            if not isinstance(point, dict):
                errors.append(f"points[{idx}] 不是对象")
                continue

            point_type = point.get("type")
            if not point_type:
                errors.append(f"points[{idx}] 缺少 type 字段")
                continue

            type_stats[point_type] = type_stats.get(point_type, 0) + 1

            if point_type not in supported_types:
                warnings.append(f"points[{idx}] 不支持的 type: {point_type}")

            if point_type == "ServoJ":
                if not in_servo_segment:
                    servo_segment_count += 1
                    in_servo_segment = True
                    current_segment_joint_dim = None

                joint_pose = point.get("joint_pose")
                if not isinstance(joint_pose, list) or not joint_pose:
                    errors.append(f"points[{idx}] ServoJ 缺少有效 joint_pose")
                    continue

                dim = len(joint_pose)
                if current_segment_joint_dim is None:
                    current_segment_joint_dim = dim
                elif dim != current_segment_joint_dim:
                    joint_dim_consistency = False
                    errors.append(f"points[{idx}] ServoJ joint_pose 维度不一致：期望 {current_segment_joint_dim}，实际 {dim}")
            else:
                in_servo_segment = False
                current_segment_joint_dim = None

        return ReplayAnalysisReport(
            total_points=len(points),
            point_type_stats=type_stats,
            servo_segment_count=servo_segment_count,
            joint_dim_consistency=joint_dim_consistency,
            warnings=warnings,
            errors=errors,
        )

    def prepare_replay(self, replay_id: str, options: Optional[ReplayOptions] = None) -> str:
        replay = self._get_replay(replay_id)
        if replay.state != ReplayState.LOADED:
            raise RuntimeError(f"轨迹状态异常: {replay.state.value}，请先调用 load_replay")
        report = self.analyze_replay(replay_id)
        if report.errors:
            raise RuntimeError("轨迹预检查失败: " + "; ".join(report.errors))

        normalized_options = self._normalize_options(options)
        point_count = len(replay.points)
        if point_count == 0:
            raise ValueError("空轨迹无法回放")

        start_index = 0
        end_index = point_count - 1

        session_id = str(uuid.uuid4())
        session = _ReplaySession(
            session_id=session_id,
            trajectory_id=replay_id,
            state=ReplayState.READY,
            options=normalized_options,
            start_index=start_index,
            end_index=end_index,
            producer_index=start_index,
        )
        session.running_event.set()

        with self._lock:
            self._sessions[session_id] = session
        return session_id

    def start_replay(self, session_id: str) -> bool:
        session = self._get_session(session_id)
        if session.state != ReplayState.READY:
            raise RuntimeError(f"当前状态无法 start: {session.state.value}")

        session.started_at = time.time()
        session.running_event.set()
        session.stop_event.clear()
        self._set_state(session, ReplayState.PLAYING)

        worker = threading.Thread(target=self._replay_loop, args=(session_id,), daemon=True)
        session.thread = worker
        worker.start()
        return True

    def pause_replay(self, session_id: str) -> bool:
        session = self._get_session(session_id)
        if session.state != ReplayState.PLAYING:
            raise RuntimeError(f"当前状态无法 pause: {session.state.value}")

        session.running_event.clear()
        self._set_state(session, ReplayState.PAUSED)
        return True

    def resume_replay(self, session_id: str) -> bool:
        session = self._get_session(session_id)
        if session.state != ReplayState.PAUSED:
            raise RuntimeError(f"当前状态无法 resume: {session.state.value}")

        session.running_event.set()
        self._set_state(session, ReplayState.PLAYING)
        return True

    def stop_replay(self, session_id: str) -> bool:
        session = self._get_session(session_id)
        if session.state in (ReplayState.STOPPED, ReplayState.FINISHED, ReplayState.CLOSED):
            return True

        session.stop_event.set()
        session.running_event.set()

        if session.thread and session.thread.is_alive():
            session.thread.join(timeout=1.0)

        if session.state not in (ReplayState.FINISHED, ReplayState.ERROR, ReplayState.CLOSED):
            session.ended_at = time.time()
            self._set_state(session, ReplayState.STOPPED)
        return True

    def step_replay(self, session_id: str, count: int = 1) -> int:
        session = self._get_session(session_id)
        if count < 1:
            raise ValueError("count 必须大于 0")
        if session.state not in (ReplayState.READY, ReplayState.PAUSED):
            raise RuntimeError(f"当前状态无法 step: {session.state.value}")

        replay = self._get_replay(session.trajectory_id)

        for _ in range(count):
            if session.producer_index > session.end_index:
                session.ended_at = time.time()
                self._set_state(session, ReplayState.FINISHED)
                break

            with session._exec_lock:
                point = replay.points[session.producer_index]
                self._execute_single_point(session, point)
                session.producer_index += 1

        if session.producer_index > session.end_index and session.state not in (ReplayState.FINISHED, ReplayState.CLOSED):
            session.ended_at = time.time()
            self._set_state(session, ReplayState.FINISHED)
        return session.producer_index

    def get_replay_status(self, session_id: str) -> ReplayStatus:
        session = self._get_session(session_id)
        progress = self._session_progress(session)

        eta_seconds: Optional[float] = None
        if session.started_at and session.state in (ReplayState.PLAYING, ReplayState.PAUSED):
            elapsed = max(0.001, time.time() - session.started_at)
            if progress > 0:
                eta_seconds = max(0.0, elapsed * (1.0 - progress) / progress)

        return ReplayStatus(
            session_id=session.session_id,
            state=session.state,
            producer_index=session.producer_index,
            consumed_index=session.consumed_index,
            progress=progress,
            eta_seconds=eta_seconds,
            last_error=session.last_error,
        )

    def close_replay(self, session_id: str) -> bool:
        session = self._get_session(session_id)
        self.stop_replay(session_id)
        self._set_state(session, ReplayState.CLOSED)

        with self._lock:
            self._sessions.pop(session_id, None)
        return True

    def subscribe_replay_events(self, session_id: str, callback: ReplayEventCallback, event_types: Optional[List[str]] = None) -> str:
        session = self._get_session(session_id)
        if event_types is not None and len(event_types) == 0:
            raise ValueError("event_types 不能为空列表")

        subscription_id = str(uuid.uuid4())
        session.subscription_callbacks[subscription_id] = _ReplayEventSubscription(
            callback=callback,
            event_types=set(event_types) if event_types is not None else None,
        )
        return subscription_id

    def unsubscribe_replay_events(self, session_id: str, subscription_id: str) -> bool:
        session = self._get_session(session_id)
        return session.subscription_callbacks.pop(subscription_id, None) is not None
