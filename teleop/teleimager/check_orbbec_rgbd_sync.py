#!/usr/bin/env python3
import argparse
import copy
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import yaml


THIS_DIR = Path(__file__).resolve().parent
SRC_DIR = THIS_DIR / "src"
REPO_ROOT = THIS_DIR.parents[1]
for path in (str(SRC_DIR), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from teleimager.image_server import OrbbecCamera


CAMERA_KEYS = ("head_camera", "left_wrist_camera", "right_wrist_camera")


def _load_rgbd_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config = copy.deepcopy(config)
    for key in CAMERA_KEYS:
        if key not in config:
            raise KeyError(f"Missing camera config: {key}")
        cam = config[key]
        cam["type"] = "orbbec"
        cam["enable_zmq"] = True
        cam["enable_depth"] = True
        cam["enable_webrtc"] = False
    return {key: config[key] for key in CAMERA_KEYS}


def _frame_status(frame):
    if frame is None:
        return False, False, None
    has_rgb = getattr(frame, "bgr", None) is not None
    has_depth = getattr(frame, "depth", None) is not None
    return has_rgb, has_depth, getattr(frame, "timestamp_ns", None)


class CameraWorker:
    def __init__(self, name: str, config: dict):
        self.name = name
        self.config = config
        self.shape = tuple(config["image_shape"])
        self.camera = OrbbecCamera(
            name,
            str(config["serial_number"]),
            list(self.shape),
            int(config.get("fps", 30)),
            enable_zmq=False,
            enable_webrtc=True,
            enable_depth=True,
        )
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._frame = None
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def _run(self):
        period = 1.0 / max(1, int(self.config.get("fps", 30)))
        while not self._stop.is_set():
            start = time.monotonic()
            try:
                self.camera._update_frame()
                depth_bytes = self.camera.get_depth_frame()
                depth = None
                if depth_bytes is not None:
                    depth = np.frombuffer(depth_bytes, dtype=np.uint16).reshape(self.shape)
                frame = SimpleNamespace(
                    bgr=self.camera.get_bgr_frame(),
                    depth=depth,
                    fps=float(self.config.get("fps", 30)),
                    timestamp_ns=self.camera.get_last_timestamp_ns(),
                )
                with self._lock:
                    self._frame = frame
            except Exception as exc:
                print(f"{self.name}: capture failed: {exc}", flush=True)
                self._stop.set()
                break
            sleep_time = period - (time.monotonic() - start)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def frame(self):
        with self._lock:
            return self._frame

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2.0)
        self.camera.release()


def _save_snapshot(save_dir: Path, frames: dict, idx: int):
    save_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in frames.items():
        if frame is None:
            continue
        if frame.bgr is not None:
            cv2.imwrite(str(save_dir / f"{idx:04d}_{name}_rgb.jpg"), frame.bgr)
        if frame.depth is not None:
            cv2.imwrite(str(save_dir / f"{idx:04d}_{name}_depth.png"), frame.depth)


def main():
    parser = argparse.ArgumentParser(
        description="Start three Orbbec cameras and monitor RGBD stream sync."
    )
    parser.add_argument(
        "--config",
        default=str(THIS_DIR / "cam_config_server_real.yaml"),
        help="Camera config path.",
    )
    parser.add_argument("--duration", type=float, default=30.0, help="Run seconds; <=0 means forever.")
    parser.add_argument("--show", action="store_true", help="Show RGB and depth preview windows.")
    parser.add_argument("--save-dir", default=None, help="Optional directory for periodic RGBD snapshots.")
    parser.add_argument("--save-every", type=float, default=5.0, help="Snapshot interval in seconds.")
    args = parser.parse_args()

    config = _load_rgbd_config(Path(args.config))
    print("Starting three Orbbec RGBD cameras:")
    for key in CAMERA_KEYS:
        cam = config[key]
        print(f"  {key}: serial={cam.get('serial_number')} port={cam.get('zmq_port')} shape={cam.get('image_shape')}")

    workers = {}
    save_dir = Path(args.save_dir) if args.save_dir else None
    last_print = 0.0
    last_save = 0.0
    save_idx = 0
    try:
        for key in CAMERA_KEYS:
            short_name = key.replace("_camera", "")
            workers[short_name] = CameraWorker(short_name, config[key])
            workers[short_name].start()
        time.sleep(1.0)
        start = time.monotonic()

        while args.duration <= 0 or time.monotonic() - start < args.duration:
            frames = {name: worker.frame() for name, worker in workers.items()}
            now = time.monotonic()
            if now - last_print >= 1.0:
                last_print = now
                timestamps = []
                parts = []
                for name, frame in frames.items():
                    has_rgb, has_depth, ts = _frame_status(frame)
                    if ts is not None:
                        timestamps.append(ts)
                    fps = frame.fps if frame is not None else 0.0
                    rgb_shape = None if not has_rgb else tuple(frame.bgr.shape)
                    depth_shape = None if not has_depth else tuple(frame.depth.shape)
                    parts.append(
                        f"{name}: fps={fps:5.1f} rgb={has_rgb} {rgb_shape} "
                        f"depth={has_depth} {depth_shape}"
                    )
                skew_ms = None
                if len(timestamps) == len(frames):
                    skew_ms = (max(timestamps) - min(timestamps)) / 1_000_000.0
                skew_text = "n/a" if skew_ms is None else f"{skew_ms:.2f} ms"
                print(f"sync_skew={skew_text} | " + " | ".join(parts), flush=True)

            if args.show:
                for name, frame in frames.items():
                    if frame is None:
                        continue
                    if frame.bgr is not None:
                        cv2.imshow(f"{name} rgb", frame.bgr)
                    if frame.depth is not None:
                        depth_vis = cv2.convertScaleAbs(frame.depth, alpha=0.03)
                        depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
                        cv2.imshow(f"{name} depth", depth_vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if save_dir is not None and now - last_save >= args.save_every:
                last_save = now
                _save_snapshot(save_dir, frames, save_idx)
                print(f"saved snapshot #{save_idx} to {save_dir}", flush=True)
                save_idx += 1

            time.sleep(0.005)
    finally:
        for worker in workers.values():
            worker.stop()
        if args.show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
