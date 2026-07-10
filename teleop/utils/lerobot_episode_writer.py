from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

import logging_mp

logger_mp = logging_mp.getLogger(__name__)

STATE_GROUPS = ("left_arm", "right_arm", "left_ee", "right_ee", "body")

# Physical cameras in a fixed order. The dataset key for each stream is derived
# from this order (see build_camera_key_plan), NOT from how many frames happen
# to be present on a given control cycle — so a dropped frame can never renumber
# another camera's stream.
CAMERA_SOURCES = (
    ("head_camera", "head"),
    ("left_wrist_camera", "left_wrist"),
    ("right_wrist_camera", "right_wrist"),
)


def build_camera_key_plan(camera_config: Any) -> list[dict[str, Any]]:
    """Map each enabled physical camera to a FIXED dataset key.

    Returns an ordered list of entries::

        {"cam": "head_camera", "source": "head", "half": None | 0 | 1,
         "color_key": "color_0", "depth_key": "depth_0" | None}

    ``half`` handles a binocular head camera whose single frame is split into a
    left/right pair. Keys are numbered by config order and stay stable for the
    whole session, so a camera that misses a frame never shifts another
    camera's ``color_N`` / ``depth_N`` index.
    """
    plan: list[dict[str, Any]] = []
    idx = 0
    for cam_name, source in CAMERA_SOURCES:
        cam = (camera_config or {}).get(cam_name) or {}
        if not cam.get("enable_zmq", False):
            continue
        enable_depth = bool(cam.get("enable_depth", False))
        halves = (0, 1) if cam.get("binocular", False) else (None,)
        for half in halves:
            plan.append(
                {
                    "cam": cam_name,
                    "source": source,
                    "half": half,
                    "color_key": f"color_{idx}",
                    "depth_key": f"depth_{idx}" if enable_depth else None,
                }
            )
            idx += 1
    return plan


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
        expected_color_keys: Any | None = None,
        expected_depth_keys: Any | None = None,
        schema_warmup_frames: int = 90,
        color_input_bgr: bool = True,
        vcodec: str = "h264",
    ):
        self.root = Path(root)
        self.repo_id = repo_id or default_repo_id(self.root.name)
        self.fps = int(round(float(fps))) if fps else 30
        self.task = task
        self.robot_type = robot_type
        self.tolerance_s = float(tolerance_s)
        self.use_videos = use_videos
        self.dataset_cls = dataset_cls
        # Camera frames arrive from OpenCV as BGR; LeRobot expects RGB, so
        # flip channels on write unless the caller already provides RGB.
        self.color_input_bgr = bool(color_input_bgr)
        # Video codec for LeRobot's encoder. LeRobot 0.4.x defaults to
        # 'libsvtav1' (AV1), which many players/browsers can't decode. 'h264'
        # is broadly playable and lossy-equivalent for training.
        self.vcodec = vcodec
        # Full set of camera stream keys that SHOULD appear once every enabled
        # camera has produced a frame. The dataset schema is not committed until
        # all of these are seen (or the warmup budget is exhausted), so a camera
        # that is slow to start never gets dropped from the schema.
        self.expected_color_keys = list(expected_color_keys or [])
        self.expected_depth_keys = list(expected_depth_keys or [])
        self.schema_warmup_frames = int(schema_warmup_frames)

        self.dataset = None
        self.episode_active = False
        self.item_id = -1
        self.current_episode_index = 0
        self.pending_alignment: list[dict[str, Any]] = []
        self.feature_specs: dict[str, Any] = {}
        self.last_feature_values: dict[str, np.ndarray] = {}
        self.closed = False
        # Frames buffered before the schema is committed, plus the merged view
        # of every camera stream seen so far, used to build a complete schema.
        self._warmup_samples: list[dict[str, Any]] = []
        self._seen_colors: dict[str, Any] = {}
        self._seen_depths: dict[str, Any] = {}

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
        colors = colors or {}
        depths = depths or {}

        # Resuming an existing dataset: the schema is already fixed on disk, so
        # there is nothing to warm up — emit straight away.
        if self.dataset is None and self._has_lerobot_metadata():
            self.dataset = self._create_or_resume_dataset(colors, depths, states, actions)
            self.current_episode_index = self._next_episode_index()
            self._capture_dataset_features()

        sample = {
            "colors": colors,
            "depths": depths,
            "states": states,
            "actions": actions,
            "frame_index": self.item_id,
            "alignment": alignment or {},
        }

        if self.dataset is None:
            # Schema not committed yet: remember every camera stream we have seen
            # and buffer the frame. Once all enabled cameras have produced at
            # least one frame (or the warmup budget runs out) we build a schema
            # that covers all of them, then flush the buffer.
            for key, image in colors.items():
                self._seen_colors.setdefault(key, image)
            for key, depth in depths.items():
                self._seen_depths.setdefault(key, depth)
            self._warmup_samples.append(sample)
            if self._schema_ready():
                self._commit_schema_and_flush(states, actions)
            return

        self._emit_frame(sample)

    def _schema_ready(self) -> bool:
        """True once we have seen every expected camera stream, or the warmup
        budget is exhausted (so a camera that never starts can't stall forever)."""
        expected_colors = set(self.expected_color_keys)
        expected_depths = set(self.expected_depth_keys)
        have_all = expected_colors.issubset(self._seen_colors) and expected_depths.issubset(
            self._seen_depths
        )
        if have_all:
            return True
        if len(self._warmup_samples) >= self.schema_warmup_frames:
            missing_c = sorted(expected_colors - set(self._seen_colors))
            missing_d = sorted(expected_depths - set(self._seen_depths))
            if missing_c or missing_d:
                logger_mp.warning(
                    "[LeRobotEpisodeWriter] Committing schema after %d warmup frames; "
                    "cameras color=%s depth=%s never produced a frame and are "
                    "EXCLUDED from the dataset. Check that these cameras are "
                    "connected and streaming before recording.",
                    len(self._warmup_samples),
                    missing_c,
                    missing_d,
                )
            return True
        return False

    def _commit_schema_and_flush(self, states, actions):
        # Build the schema from the union of every camera seen during warmup so
        # no enabled stream is missing, then replay the buffered frames.
        self.dataset = self._create_or_resume_dataset(
            self._seen_colors, self._seen_depths, states, actions
        )
        self.current_episode_index = self._next_episode_index()
        self._capture_dataset_features()
        buffered = self._warmup_samples
        self._warmup_samples = []
        for sample in buffered:
            self._emit_frame(sample)

    def _emit_frame(self, sample):
        frame = self._build_lerobot_frame(
            sample["colors"], sample["depths"], sample["states"], sample["actions"]
        )
        frame = self._complete_frame(frame)
        self.dataset.add_frame(frame)
        self._remember_frame_values(frame)
        self.pending_alignment.append(
            {
                "episode_index": self.current_episode_index,
                "frame_index": sample["frame_index"],
                "timestamp": sample["frame_index"] / self.fps if self.fps > 0 else 0.0,
                "source_idx": sample["frame_index"],
                "alignment": sample["alignment"],
            }
        )

    def save_episode(self):
        if not self.episode_active:
            return
        # A very short episode may end before every camera showed up. Commit the
        # schema from whatever we have and flush the buffer so no frames are lost.
        if self.dataset is None and self._warmup_samples:
            last = self._warmup_samples[-1]
            self._commit_schema_and_flush(last["states"], last["actions"])
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
            dataset = dataset_cls.resume(
                repo_id=self.repo_id,
                root=self.root,
                tolerance_s=self.tolerance_s,
            )
            self.feature_specs = dict(getattr(dataset, "features", {}) or {})
            return dataset

        if self.root.exists():
            if any(self.root.iterdir()):
                raise RuntimeError(
                    f"LeRobot output directory is not empty and has no meta/info.json: {self.root}. "
                    "Choose a new --task-name/--task-dir or remove the old legacy recording directory."
                )
            self.root.rmdir()

        features = self._build_features(colors, depths or {}, states, actions)
        self.feature_specs = features
        create_kwargs = dict(
            repo_id=self.repo_id,
            fps=self.fps,
            features=features,
            root=self.root,
            robot_type=self.robot_type,
            use_videos=self.use_videos,
            tolerance_s=self.tolerance_s,
        )
        # Only request a specific codec when actually encoding videos.
        if self.use_videos and self.vcodec:
            create_kwargs["vcodec"] = self.vcodec
        return dataset_cls.create(**create_kwargs)

    def _capture_dataset_features(self):
        if self.dataset is None:
            return
        dataset_features = getattr(self.dataset, "features", None)
        if dataset_features:
            self.feature_specs = dict(dataset_features)

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
        # Only emit camera streams that are part of the committed schema. A
        # camera whose first frame arrived AFTER the schema was committed (its
        # shape was never known at build time) is dropped rather than passed to
        # add_frame, which would reject the unknown key and crash recording.
        for key, image in sorted((colors or {}).items()):
            spec_key = f"observation.images.{key}"
            if self._key_in_schema(spec_key):
                arr = np.asarray(image)
                # LeRobot stores images as RGB. Camera frames come from OpenCV
                # as BGR, so flip the channel order (matches the offline
                # export_lerobot_dataset.py read_rgb path). Without this the
                # recorded videos have red/blue channels swapped.
                if self.color_input_bgr and arr.ndim == 3 and arr.shape[-1] == 3:
                    arr = arr[..., ::-1]
                frame[spec_key] = np.ascontiguousarray(arr)
        for key, depth in sorted((depths or {}).items()):
            spec_key = f"observation.depths.{key}"
            if self._key_in_schema(spec_key):
                frame[spec_key] = depth_as_rgb(depth)
        return frame

    def _key_in_schema(self, spec_key: str) -> bool:
        # Before the schema is committed feature_specs is empty; accept everything
        # so warmup can collect shapes. After commit, restrict to known keys.
        if not self.feature_specs:
            return True
        return spec_key in self.feature_specs

    def _complete_frame(self, frame):
        for key, spec in (self.feature_specs or {}).items():
            if key in frame:
                continue
            if key in self.last_feature_values:
                frame[key] = self.last_feature_values[key].copy()
                continue
            if key.startswith(("observation.images.", "observation.depths.")):
                shape = tuple(spec.get("shape", ())) if isinstance(spec, dict) else ()
                if len(shape) == 3:
                    frame[key] = np.zeros(shape, dtype=np.uint8)
                    continue
            if key in ("observation.state", "action"):
                shape = tuple(spec.get("shape", ())) if isinstance(spec, dict) else ()
                frame[key] = np.zeros(shape, dtype=np.float32)
        return frame

    def _remember_frame_values(self, frame):
        for key, value in frame.items():
            if key.startswith(("observation.images.", "observation.depths.")):
                self.last_feature_values[key] = np.asarray(value).copy()

    def _write_alignment_records(self, records):
        if not records:
            return
        sidecar_path = self.root / "meta" / "tele_robot_alignment.jsonl"
        sidecar_path.parent.mkdir(parents=True, exist_ok=True)
        with sidecar_path.open("a", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
