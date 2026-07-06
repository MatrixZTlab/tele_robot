from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

STATE_GROUPS = ("left_arm", "right_arm", "left_ee", "right_ee", "body")


def flatten_numeric_tree(tree: Any, prefix: str = "") -> tuple[list[float], list[str]]:
    values: list[float] = []
    names: list[str] = []

    if tree is None:
        return values, names

    if isinstance(tree, dict):
        ordered_keys = [key for key in STATE_GROUPS if key in tree]
        ordered_keys += sorted(key for key in tree.keys() if key not in ordered_keys)
        for key in ordered_keys:
            child_prefix = f"{prefix}_{key}" if prefix else str(key)
            child_values, child_names = flatten_numeric_tree(tree[key], child_prefix)
            values.extend(child_values)
            names.extend(child_names)
        return values, names

    if isinstance(tree, (list, tuple, np.ndarray)):
        arr = np.asarray(tree).reshape(-1)
        for i, value in enumerate(arr):
            values.append(float(value))
            names.append(f"{prefix}_{i}" if prefix else str(i))
        return values, names

    if isinstance(tree, (int, float, np.number, bool)):
        values.append(float(tree))
        names.append(prefix or "value")

    return values, names


def depth_as_rgb(depth: Any) -> np.ndarray:
    depth_array = np.asarray(depth)
    if depth_array.ndim == 3:
        depth_array = depth_array[:, :, 0]

    depth_f = depth_array.astype(np.float32)
    valid = depth_f[np.isfinite(depth_f) & (depth_f > 0)]
    if valid.size == 0:
        depth_u8 = np.zeros(depth_f.shape, dtype=np.uint8)
    else:
        lo = float(np.percentile(valid, 1.0))
        hi = float(np.percentile(valid, 99.0))
        if hi <= lo:
            hi = lo + 1.0
        depth_u8 = np.clip((depth_f - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)

    return np.repeat(depth_u8[:, :, None], 3, axis=2)


def default_repo_id(task_name: str) -> str:
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", task_name.strip()).strip("_")
    return f"local/{safe_name or 'tele_robot'}"


def load_lerobot_dataset_cls():
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:
        raise RuntimeError(
            "LeRobotDataset recording requires the lerobot package in this Python environment. "
            "Install/run with the LeRobot environment before using --record."
        ) from exc
    return LeRobotDataset


class LeRobotEpisodeWriter:
    """Write tele_robot samples directly into a LeRobotDataset.

    The class intentionally mirrors the old EpisodeWriter lifecycle used by
    teleop_hand_and_arm.py: create_episode(), add_item(), save_episode(), close().
    """

    def __init__(
        self,
        root: str | Path,
        repo_id: str | None = None,
        fps: float = 30.0,
        task: str = "task",
        robot_type: str = "topstar",
        tolerance_s: float = 1e-4,
        use_videos: bool = True,
        dataset_cls: Any | None = None,
    ):
        self.root = Path(root)
        self.repo_id = repo_id or default_repo_id(self.root.name)
        self.fps = int(round(float(fps))) if fps else 30
        self.task = task
        self.robot_type = robot_type
        self.tolerance_s = float(tolerance_s)
        self.use_videos = use_videos
        self.dataset_cls = dataset_cls

        self.dataset = None
        self.episode_active = False
        self.item_id = -1
        self.current_episode_index = 0
        self.pending_alignment: list[dict[str, Any]] = []
        self.closed = False

    def is_ready(self):
        return not self.episode_active

    def create_episode(self):
        if self.episode_active:
            return False
        self.item_id = -1
        self.pending_alignment = []
        self.current_episode_index = self._next_episode_index()
        self.episode_active = True
        return True

    def add_item(
        self,
        colors,
        depths=None,
        states=None,
        actions=None,
        tactiles=None,
        audios=None,
        sim_state=None,
        alignment=None,
    ):
        if not self.episode_active:
            if not self.create_episode():
                raise RuntimeError("Could not create a new LeRobot episode")

        self.item_id += 1
        if self.dataset is None:
            self.dataset = self._create_or_resume_dataset(colors, depths, states, actions)
            self.current_episode_index = self._next_episode_index()

        frame = self._build_lerobot_frame(colors, depths or {}, states, actions)
        self.dataset.add_frame(frame)
        self.pending_alignment.append(
            {
                "episode_index": self.current_episode_index,
                "frame_index": self.item_id,
                "timestamp": self.item_id / self.fps if self.fps > 0 else 0.0,
                "source_idx": self.item_id,
                "alignment": alignment or {},
            }
        )

    def save_episode(self):
        if not self.episode_active:
            return
        if self.dataset is not None and self.item_id >= 0:
            self.dataset.save_episode()
            self._write_alignment_records(self.pending_alignment)
        self.pending_alignment = []
        self.episode_active = False

    def close(self):
        if self.closed:
            return
        if self.episode_active:
            self.save_episode()
        if self.dataset is not None:
            self.dataset.finalize()
        self.closed = True

    def _create_or_resume_dataset(self, colors, depths, states, actions):
        dataset_cls = self.dataset_cls or load_lerobot_dataset_cls()
        if self._has_lerobot_metadata():
            return dataset_cls.resume(
                repo_id=self.repo_id,
                root=self.root,
                tolerance_s=self.tolerance_s,
            )

        if self.root.exists():
            if any(self.root.iterdir()):
                raise RuntimeError(
                    f"LeRobot output directory is not empty and has no meta/info.json: {self.root}. "
                    "Choose a new --task-name/--task-dir or remove the old legacy recording directory."
                )
            self.root.rmdir()

        features = self._build_features(colors, depths or {}, states, actions)
        return dataset_cls.create(
            repo_id=self.repo_id,
            fps=self.fps,
            features=features,
            root=self.root,
            robot_type=self.robot_type,
            use_videos=self.use_videos,
            tolerance_s=self.tolerance_s,
        )

    def _has_lerobot_metadata(self):
        return (self.root / "meta" / "info.json").exists()

    def _next_episode_index(self):
        if self.dataset is None:
            return 0
        return int(getattr(self.dataset, "num_episodes", 0))

    def _build_features(self, colors, depths, states, actions):
        state_values, state_names = flatten_numeric_tree(states)
        action_values, action_names = flatten_numeric_tree(actions)
        if not state_values:
            raise ValueError("Cannot create LeRobotDataset: states are empty")
        if not action_values:
            raise ValueError("Cannot create LeRobotDataset: actions are empty")

        features: dict[str, Any] = {
            "observation.state": {
                "dtype": "float32",
                "shape": (len(state_values),),
                "names": {"axes": state_names},
            },
            "action": {
                "dtype": "float32",
                "shape": (len(action_values),),
                "names": {"axes": action_names},
            },
        }
        image_dtype = "video" if self.use_videos else "image"
        for key, image in sorted((colors or {}).items()):
            features[f"observation.images.{key}"] = {
                "dtype": image_dtype,
                "shape": tuple(np.asarray(image).shape),
                "names": ["height", "width", "channels"],
            }
        for key, depth in sorted((depths or {}).items()):
            features[f"observation.depths.{key}"] = {
                "dtype": image_dtype,
                "shape": tuple(depth_as_rgb(depth).shape),
                "names": ["height", "width", "channels"],
            }
        return features

    def _build_lerobot_frame(self, colors, depths, states, actions):
        state_values, _ = flatten_numeric_tree(states)
        action_values, _ = flatten_numeric_tree(actions)
        frame: dict[str, Any] = {
            "task": self.task,
            "observation.state": np.asarray(state_values, dtype=np.float32),
            "action": np.asarray(action_values, dtype=np.float32),
        }
        for key, image in sorted((colors or {}).items()):
            frame[f"observation.images.{key}"] = np.asarray(image)
        for key, depth in sorted((depths or {}).items()):
            frame[f"observation.depths.{key}"] = depth_as_rgb(depth)
        return frame

    def _write_alignment_records(self, records):
        if not records:
            return
        sidecar_path = self.root / "meta" / "tele_robot_alignment.jsonl"
        sidecar_path.parent.mkdir(parents=True, exist_ok=True)
        with sidecar_path.open("a", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
