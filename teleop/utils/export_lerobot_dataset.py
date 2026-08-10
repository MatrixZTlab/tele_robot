#!/usr/bin/env python3
"""Export tele_robot EpisodeWriter data to a standard LeRobot dataset."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except ImportError as exc:  # pragma: no cover - depends on local environment
    raise SystemExit(
        "Could not import lerobot. Run this script with the LeRobot environment, "
        "for example:\n"
        "  /home/top/miniforge3/envs/lerobot4/bin/python "
        "teleop/utils/export_lerobot_dataset.py --input-dir teleop/utils/data"
    ) from exc


STATE_GROUPS = ("left_arm", "right_arm", "left_ee", "right_ee", "body")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("teleop/utils/data"),
        help="Directory containing episode_XXXX folders, or parent task directories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("teleop/utils/lerobot_data/tele_robot"),
        help="Output LeRobot dataset root.",
    )
    parser.add_argument("--repo-id", default="local/tele_robot", help="LeRobot repo id metadata.")
    parser.add_argument("--robot-type", default="topstar", help="Robot type metadata.")
    parser.add_argument("--task-name", default=None, help="Only export episodes under this task directory.")
    parser.add_argument("--fps", type=int, default=None, help="Override FPS. Defaults to source info.fps.")
    parser.add_argument("--tolerance-s", type=float, default=None, help="Override LeRobot timestamp tolerance.")
    parser.add_argument("--no-depth", action="store_true", help="Skip depth streams.")
    parser.add_argument("--no-videos", action="store_true", help="Store image features as images instead of videos.")
    parser.add_argument(
        "--streaming-encoding",
        action="store_true",
        help="Encode video frames directly instead of staging temporary PNG files.",
    )
    parser.add_argument(
        "--vcodec",
        default="libsvtav1",
        help="LeRobot video codec, for example libsvtav1, h264, or h264_nvenc.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Delete output directory before exporting.")
    parser.add_argument(
        "--allow-invalid-alignment",
        action="store_true",
        help="Export aligned episodes even when alignment_report.json marks them invalid.",
    )
    return parser.parse_args()


def find_episode_dirs(input_dir: Path, task_name: str | None) -> list[Path]:
    root = input_dir.expanduser().resolve()
    if task_name:
        root = root / task_name

    if not root.exists():
        raise FileNotFoundError(f"Input directory does not exist: {root}")

    if root.name.startswith("episode_") and (root / "data.json").exists():
        return [root]

    episode_dirs = sorted(p for p in root.rglob("episode_*") if (p / "data.json").exists())
    if not episode_dirs:
        raise FileNotFoundError(f"No episode_XXXX/data.json files found under {root}")
    return episode_dirs


def load_episode(episode_dir: Path) -> dict[str, Any]:
    with (episode_dir / "data.json").open("r", encoding="utf-8") as f:
        episode = json.load(f)
    if not episode.get("data"):
        raise ValueError(f"Episode has no frames: {episode_dir}")
    return episode


def validate_alignment_report(episode_dir: Path, allow_invalid: bool) -> None:
    report_path = episode_dir / "alignment_report.json"
    if allow_invalid or not report_path.exists():
        return
    report = json.loads(report_path.read_text(encoding="utf-8"))
    failures = []
    if report.get("quality_valid") is False:
        failures.extend(report.get("quality_issues") or ["quality_invalid"])
    if report.get("source_clock_valid") is False:
        failures.extend(report.get("source_clock_issues") or ["source_clock_invalid"])
    if failures:
        raise ValueError(
            f"Refusing to export invalid aligned episode {episode_dir}: "
            + ", ".join(str(item) for item in failures)
        )


def flatten_numeric_tree(tree: Any, prefix: str = "") -> tuple[list[float], list[str]]:
    values: list[float] = []
    names: list[str] = []

    if tree is None:
        return values, names

    if isinstance(tree, dict):
        ordered_keys = [key for key in STATE_GROUPS if key in tree]
        ordered_keys += sorted(key for key in tree.keys() if key not in ordered_keys)
        for key in ordered_keys:
            child = tree[key]
            child_prefix = f"{prefix}_{key}" if prefix else str(key)
            child_values, child_names = flatten_numeric_tree(child, child_prefix)
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


def read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def read_depth_as_rgb(path: Path) -> np.ndarray:
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(f"Could not read depth image: {path}")

    if depth.ndim == 3:
        depth = depth[:, :, 0]

    depth_f = depth.astype(np.float32)
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


def first_frame_with_key(episodes: list[tuple[Path, dict[str, Any]]], key: str) -> tuple[Path, dict[str, Any]] | None:
    for episode_dir, episode in episodes:
        for frame in episode["data"]:
            if frame.get(key):
                return episode_dir, frame
    return None


def build_features(
    episodes: list[tuple[Path, dict[str, Any]]],
    use_depth: bool,
    use_videos: bool,
) -> tuple[dict[str, Any], list[str], list[str], list[str], list[str]]:
    sample_frame = episodes[0][1]["data"][0]
    state_values, state_names = flatten_numeric_tree(sample_frame.get("states"))
    action_values, action_names = flatten_numeric_tree(sample_frame.get("actions"))

    if not state_values:
        raise ValueError("First frame has empty states; cannot create observation.state feature.")
    if not action_values:
        raise ValueError("First frame has empty actions; cannot create action feature.")

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

    image_dtype = "video" if use_videos else "image"
    color_sample = first_frame_with_key(episodes, "colors")
    depth_sample = first_frame_with_key(episodes, "depths") if use_depth else None
    color_keys = sorted((color_sample[1].get("colors") or {}).keys()) if color_sample else []
    depth_keys = sorted((depth_sample[1].get("depths") or {}).keys()) if depth_sample else []

    for color_key in color_keys:
        color_episode_dir, color_frame = color_sample
        rel_path = color_frame["colors"][color_key]
        image = read_rgb(color_episode_dir / rel_path)
        features[f"observation.images.{color_key}"] = {
            "dtype": image_dtype,
            "shape": tuple(image.shape),
            "names": ["height", "width", "channels"],
        }

    for depth_key in depth_keys:
        depth_episode_dir, depth_frame = depth_sample
        rel_path = depth_frame["depths"][depth_key]
        depth_image = read_depth_as_rgb(depth_episode_dir / rel_path)
        features[f"observation.depths.{depth_key}"] = {
            "dtype": image_dtype,
            "shape": tuple(depth_image.shape),
            "names": ["height", "width", "channels"],
        }

    return features, state_names, action_names, color_keys, depth_keys


def source_fps_and_tolerance(episodes: list[tuple[Path, dict[str, Any]]]) -> tuple[int, float]:
    info = episodes[0][1].get("info") or {}
    fps = int(round(float(info.get("fps", 30))))
    tolerance_s = float(info.get("tolerance_s", 1e-4))
    return fps, tolerance_s


def source_task_text(episode: dict[str, Any], fallback: str) -> str:
    text = episode.get("text") or {}
    return text.get("goal") or text.get("desc") or fallback


def vector_from_frame(frame: dict[str, Any], key: str, expected_dim: int, episode_dir: Path) -> np.ndarray:
    values, _ = flatten_numeric_tree(frame.get(key))
    if len(values) != expected_dim:
        raise ValueError(
            f"{episode_dir.name} frame {frame.get('idx', frame.get('frame_index'))} "
            f"has {len(values)} {key} values, expected {expected_dim}"
        )
    return np.asarray(values, dtype=np.float32)


def alignment_record(episode_index: int, frame_index: int, frame: dict[str, Any]) -> dict[str, Any]:
    alignment = frame.get("alignment") or {}
    return {
        "episode_index": episode_index,
        "frame_index": frame_index,
        "timestamp": frame.get("timestamp"),
        "source_idx": frame.get("idx"),
        "alignment": alignment,
    }


def write_dataset(args: argparse.Namespace) -> Path:
    episode_dirs = find_episode_dirs(args.input_dir, args.task_name)
    for episode_dir in episode_dirs:
        validate_alignment_report(episode_dir, args.allow_invalid_alignment)
    episodes = [(episode_dir, load_episode(episode_dir)) for episode_dir in episode_dirs]
    fps, tolerance_s = source_fps_and_tolerance(episodes)
    fps = args.fps if args.fps is not None else fps
    tolerance_s = args.tolerance_s if args.tolerance_s is not None else tolerance_s

    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output directory exists. Use --overwrite to replace it: {output_dir}")
        shutil.rmtree(output_dir)

    features, state_names, action_names, color_keys, depth_keys = build_features(
        episodes,
        use_depth=not args.no_depth,
        use_videos=not args.no_videos,
    )

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=fps,
        features=features,
        root=output_dir,
        robot_type=args.robot_type,
        use_videos=not args.no_videos,
        tolerance_s=tolerance_s,
        streaming_encoding=args.streaming_encoding,
        vcodec=args.vcodec,
    )

    alignment_records: list[dict[str, Any]] = []
    for episode_index, (episode_dir, episode) in enumerate(episodes):
        task = source_task_text(episode, episode_dir.parent.name)
        for local_frame_index, frame in enumerate(episode["data"]):
            lerobot_frame: dict[str, Any] = {
                "task": task,
                "observation.state": vector_from_frame(frame, "states", len(state_names), episode_dir),
                "action": vector_from_frame(frame, "actions", len(action_names), episode_dir),
            }

            colors = frame.get("colors") or {}
            for color_key in color_keys:
                lerobot_frame[f"observation.images.{color_key}"] = read_rgb(episode_dir / colors[color_key])

            depths = frame.get("depths") or {}
            for depth_key in depth_keys:
                lerobot_frame[f"observation.depths.{depth_key}"] = read_depth_as_rgb(episode_dir / depths[depth_key])

            dataset.add_frame(lerobot_frame)
            alignment_records.append(alignment_record(episode_index, local_frame_index, frame))

        dataset.save_episode()

    dataset.finalize()

    sidecar_path = output_dir / "meta" / "tele_robot_alignment.jsonl"
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    with sidecar_path.open("w", encoding="utf-8") as f:
        for record in alignment_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary_path = output_dir / "meta" / "tele_robot_export.json"
    summary = {
        "source_input_dir": str(args.input_dir.expanduser().resolve()),
        "episode_count": len(episodes),
        "fps": fps,
        "tolerance_s": tolerance_s,
        "state_dim": len(state_names),
        "action_dim": len(action_names),
        "color_keys": color_keys,
        "depth_keys": depth_keys,
        "streaming_encoding": args.streaming_encoding,
        "vcodec": args.vcodec,
        "depth_note": (
            "Depth PNGs are exported as 3-channel uint8 video frames because "
            "LeRobot 0.4.1 image writing expects 3-channel images. Raw depth stays "
            "in the source tele_robot episode."
        ),
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return output_dir


def main() -> None:
    output_dir = write_dataset(parse_args())
    print(f"Exported LeRobot dataset to: {output_dir}")


if __name__ == "__main__":
    main()
