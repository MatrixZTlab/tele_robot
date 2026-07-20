"""Asynchronous raw stream recorder for timestamp-aligned teleoperation data."""

from __future__ import annotations

import json
import os
import queue
import shutil
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from teleop.utils.check_clock_sync import clock_sync_report


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


class RawSessionWriter:
    """Write independent sensor streams without blocking their producer threads."""

    def __init__(self, task_dir: str | os.PathLike[str], queue_size: int = 8192):
        self.task_dir = Path(task_dir).expanduser().resolve()
        self.raw_root = self.task_dir / "raw"
        self.raw_root.mkdir(parents=True, exist_ok=True)
        # This subprocess-backed audit happens while the recorder is initialized,
        # before teleoperation starts. Never run it from the live control loop.
        self._control_host_clock_sync = clock_sync_report()
        self._queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self._lock = threading.Lock()
        self._session_dir: Path | None = None
        self._episode_index: int | None = None
        self._session_started_wall_ns = 0
        self._accepted = Counter()
        self._written = Counter()
        self._dropped = Counter()
        self._failed = Counter()
        self._errors: list[dict[str, Any]] = []
        self._error_lock = threading.Lock()
        self._camera_fallback_sequence = Counter()
        self._handles: dict[tuple[Path, str], Any] = {}
        self._closed = False
        self._worker = threading.Thread(target=self._run, name="raw-session-writer", daemon=True)
        self._worker.start()

    @property
    def active(self) -> bool:
        with self._lock:
            return self._session_dir is not None

    def start_episode(self, episode_index: int, metadata: dict[str, Any] | None = None) -> Path:
        with self._lock:
            if self._session_dir is not None:
                raise RuntimeError("A raw episode is already active")
            session_dir = self.raw_root / f"episode_{episode_index:04d}"
            if session_dir.exists() and any(session_dir.iterdir()):
                raise FileExistsError(f"Raw episode directory is not empty: {session_dir}")
            session_dir.mkdir(parents=True, exist_ok=True)
            (session_dir / "streams").mkdir(exist_ok=True)
            (session_dir / "cameras").mkdir(exist_ok=True)
            self._session_dir = session_dir
            self._episode_index = int(episode_index)
            self._session_started_wall_ns = time.time_ns()
            self._accepted.clear()
            self._written.clear()
            self._dropped.clear()
            self._failed.clear()
            self._camera_fallback_sequence.clear()
            with self._error_lock:
                self._errors.clear()

        manifest = {
            "format": "tele_robot_raw_v1",
            "episode_index": int(episode_index),
            "started_wall_ns": self._session_started_wall_ns,
            "clock_rules": {
                "cross_host": "wall clock synchronized by chrony/PTP",
                "local_duration": "monotonic clock",
                "pico_source": "optional; host receipt is retained when unavailable",
            },
            "control_host_clock_sync": dict(self._control_host_clock_sync),
            "metadata": metadata or {},
        }
        self._write_json_atomic(session_dir / "manifest.json", manifest)
        return session_dir

    def append_event(self, stream: str, event: dict[str, Any]) -> bool:
        with self._lock:
            session_dir = self._session_dir
            if session_dir is None or self._closed:
                return False
            return self._enqueue(("event", session_dir, str(stream), dict(event)), str(stream))

    def append_camera_packet(self, stream: str, packet: dict[str, Any]) -> bool:
        with self._lock:
            session_dir = self._session_dir
            if session_dir is None or self._closed:
                return False
            # Copy only the packet dictionary. Byte payloads are immutable and are not duplicated.
            return self._enqueue(("camera", session_dir, str(stream), dict(packet)), f"camera.{stream}")

    def stop_episode(self, timeout: float = 30.0) -> Path | None:
        with self._lock:
            session_dir = self._session_dir
            episode_index = self._episode_index
            self._session_dir = None
            self._episode_index = None
        if session_dir is None:
            return None

        barrier = threading.Event()
        self._queue.put(("barrier", session_dir, barrier), timeout=timeout)
        if not barrier.wait(timeout=timeout):
            raise TimeoutError(f"Timed out flushing raw episode: {session_dir}")

        with self._error_lock:
            errors = list(self._errors)

        manifest_path = session_dir / "manifest.json"
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        manifest.update({
            "episode_index": episode_index,
            "stopped_wall_ns": time.time_ns(),
            "accepted": dict(self._accepted),
            "written": dict(self._written),
            "dropped": dict(self._dropped),
            "failed": dict(self._failed),
            "errors": errors,
            "complete": not errors,
            "capture_valid": not errors and not any(self._dropped.values()),
        })
        self._write_json_atomic(manifest_path, manifest)
        if errors:
            summary = "; ".join(
                f"{item['kind']}: {item['error_type']}: {item['message']}"
                for item in errors[:3]
            )
            raise RuntimeError(f"Raw episode contains write failures: {summary}")
        return session_dir

    def discard_episode(
        self,
        episode_index: int | None = None,
        timeout: float = 30.0,
    ) -> Path | None:
        """Flush and permanently remove an active or completed raw episode."""
        with self._lock:
            active_index = self._episode_index

        if active_index is not None:
            if episode_index is not None and int(episode_index) != active_index:
                raise ValueError(
                    f"Active raw episode is {active_index}, not {episode_index}"
                )
            session_dir = self.raw_root / f"episode_{active_index:04d}"
            try:
                self.stop_episode(timeout=timeout)
            except (TimeoutError, queue.Full):
                raise
            except Exception:
                # The barrier has already closed stream handles before stop_episode
                # reports writer errors, so a rejected capture can still be removed.
                pass
        elif episode_index is not None:
            session_dir = self.raw_root / f"episode_{int(episode_index):04d}"
        else:
            return None

        if session_dir.exists():
            shutil.rmtree(session_dir)
            return session_dir
        return None

    def close(self, timeout: float = 30.0) -> None:
        if self._closed:
            return
        stop_error = None
        try:
            self.stop_episode(timeout=timeout)
        except Exception as exc:
            stop_error = exc
        self._closed = True
        self._queue.put(("close",), timeout=timeout)
        self._worker.join(timeout=timeout)
        if self._worker.is_alive():
            raise TimeoutError("Raw session writer did not stop")
        if stop_error is not None:
            raise stop_error

    def _enqueue(self, item: tuple, stream: str) -> bool:
        try:
            self._queue.put_nowait(item)
            self._accepted[stream] += 1
            return True
        except queue.Full:
            self._dropped[stream] += 1
            return False

    def _stream_handle(self, session_dir: Path, stream: str):
        key = (session_dir, stream)
        handle = self._handles.get(key)
        if handle is None:
            path = session_dir / "streams" / f"{stream}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            handle = path.open("a", encoding="utf-8")
            self._handles[key] = handle
        return handle

    def _write_event(self, session_dir: Path, stream: str, event: dict[str, Any]) -> None:
        handle = self._stream_handle(session_dir, stream)
        handle.write(json.dumps(_jsonable(event), ensure_ascii=False, separators=(",", ":")) + "\n")
        self._written[stream] += 1

    def _write_camera(self, session_dir: Path, stream: str, packet: dict[str, Any]) -> None:
        sequence = packet.get("sequence")
        if sequence is None:
            self._camera_fallback_sequence[stream] += 1
            sequence = self._camera_fallback_sequence[stream]
        sequence = int(sequence)
        camera_dir = session_dir / "cameras" / stream
        camera_dir.mkdir(parents=True, exist_ok=True)

        index = {key: value for key, value in packet.items() if key not in ("jpg", "depth", "bgr")}
        jpg = packet.get("jpg")
        if jpg:
            image_path = camera_dir / f"{sequence:09d}.jpg"
            image_path.write_bytes(bytes(jpg))
            index["image_path"] = str(image_path.relative_to(session_dir))

        depth = packet.get("depth")
        if depth is not None:
            depth_path = camera_dir / f"{sequence:09d}.depth"
            depth_path.write_bytes(bytes(depth))
            index["depth_path"] = str(depth_path.relative_to(session_dir))

        index["sequence"] = sequence
        self._write_event(session_dir, f"camera_{stream}", index)

    def _flush_session(self, session_dir: Path) -> None:
        for (handle_session, _), handle in list(self._handles.items()):
            if handle_session == session_dir:
                handle.flush()
                os.fsync(handle.fileno())
                handle.close()
        self._handles = {
            key: handle for key, handle in self._handles.items() if key[0] != session_dir
        }

    def _record_error(self, session_dir: Path, kind: str, exc: Exception) -> None:
        stream = kind
        self._failed[stream] += 1
        with self._error_lock:
            if len(self._errors) < 20:
                self._errors.append({
                    "wall_ns": time.time_ns(),
                    "session_dir": str(session_dir),
                    "kind": kind,
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                })

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            kind = item[0]
            barrier = item[2] if kind == "barrier" else None
            try:
                if kind == "close":
                    break
                if kind == "event":
                    _, session_dir, stream, event = item
                    self._write_event(session_dir, stream, event)
                elif kind == "camera":
                    _, session_dir, stream, packet = item
                    self._write_camera(session_dir, stream, packet)
                elif kind == "barrier":
                    _, session_dir, _ = item
                    self._flush_session(session_dir)
            except Exception as exc:
                session_dir = item[1] if len(item) > 1 else self.raw_root
                error_kind = item[2] if kind in ("event", "camera") else kind
                self._record_error(session_dir, str(error_kind), exc)
            finally:
                if barrier is not None:
                    barrier.set()
                self._queue.task_done()

        for handle in self._handles.values():
            handle.close()
        self._handles.clear()

    @staticmethod
    def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(_jsonable(payload), handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
