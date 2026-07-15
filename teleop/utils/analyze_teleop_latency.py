#!/usr/bin/env python3
"""Offline latency report for a recorded TOPSTAR LeRobot episode."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def _summary(name: str, values: np.ndarray, unit: str = "ms") -> None:
    if values.size == 0:
        print(f"{name}: no samples")
        return
    p50, p95, p99, maximum = np.percentile(values, [50, 95, 99, 100])
    print(
        f"{name}: p50={p50:.2f}{unit}, p95={p95:.2f}{unit}, "
        f"p99={p99:.2f}{unit}, max={maximum:.2f}{unit}"
    )


def _load_alignment(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    control = []
    done = []
    for row in rows:
        timestamps = row.get("alignment", {}).get("actual_timestamps_ns", {})
        pairs = (
            ("control_start_monotonic", "robot_step_done_monotonic"),
            ("control_cycle_wall", "robot_step_done_wall"),
            ("control_cycle", "robot_step_done"),
        )
        for control_key, done_key in pairs:
            if control_key in timestamps and done_key in timestamps:
                control.append(timestamps[control_key])
                done.append(timestamps[done_key])
                break
    return np.asarray(control, dtype=np.int64), np.asarray(done, dtype=np.int64)


def _best_feedback_lag_ms(
    states: np.ndarray, actions: np.ndarray, control_ns: np.ndarray, max_lag: int
) -> tuple[int, float, float]:
    best_lag = 0
    best_mae = float("inf")
    usable_max_lag = min(max_lag, len(states) - 1)
    for lag in range(usable_max_lag + 1):
        error = states[lag:] - actions[: len(states) - lag]
        mae = float(np.mean(np.abs(error)))
        if mae < best_mae:
            best_lag, best_mae = lag, mae
    if best_lag == 0 or len(control_ns) <= best_lag:
        return best_lag, 0.0, best_mae
    lag_ms = (control_ns[best_lag:] - control_ns[:-best_lag]) / 1e6
    return best_lag, float(np.median(lag_ms)), best_mae


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze recorded control timing and joint feedback lag."
    )
    parser.add_argument(
        "task_dir",
        type=Path,
        help="Task directory containing lerobot/data and lerobot/meta",
    )
    parser.add_argument("--max-lag-frames", type=int, default=12)
    args = parser.parse_args()

    root = args.task_dir.expanduser().resolve() / "lerobot"
    data_files = sorted((root / "data").glob("chunk-*/*.parquet"))
    alignment_path = root / "meta" / "tele_robot_alignment.jsonl"
    if len(data_files) != 1 or not alignment_path.is_file():
        raise SystemExit(
            "Expected exactly one LeRobot data parquet and "
            "meta/tele_robot_alignment.jsonl under " + str(root)
        )

    table = pq.read_table(data_files[0])
    states = np.asarray(table["observation.state"].to_pylist(), dtype=float)[:, :14]
    actions = np.asarray(table["action"].to_pylist(), dtype=float)[:, :14]
    control_ns, done_ns = _load_alignment(alignment_path)
    if len(control_ns) != len(states):
        raise SystemExit("Alignment row count does not match parquet frame count")

    print(f"frames: {len(states)}")
    period_ms = np.diff(control_ns) / 1e6
    step_ms = (done_ns - control_ns) / 1e6
    _summary("actual control period", period_ms)
    _summary("robot.step local duration", step_ms)
    print(f"cycles above 60ms: {int(np.sum(period_ms > 60))}/{len(period_ms)}")
    print(f"cycles above 100ms: {int(np.sum(period_ms > 100))}/{len(period_ms)}")

    lag, lag_ms, mae = _best_feedback_lag_ms(
        states, actions, control_ns, args.max_lag_frames
    )
    tracking_error = np.max(np.abs(states - actions), axis=1)
    print(
        "best state/action lag: "
        f"{lag} frames, median {lag_ms:.1f}ms, MAE={mae:.4f} rad"
    )
    _summary("current state/action max-joint error", tracking_error, " rad")
    print(
        "Note: this report cannot measure Pico-to-PC WebSocket delay because "
        "the saved trajectory has no Pico packet timestamp."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
