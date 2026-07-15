#!/usr/bin/env python3
"""Estimate LowCmd-to-LowState response delay from a raw teleoperation episode."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from teleop.utils.align_raw_session import load_jsonl


def interpolate_rows(times: np.ndarray, values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    return np.column_stack([
        np.interp(grid, times, values[:, joint]) for joint in range(values.shape[1])
    ])


def zero_order_hold(times: np.ndarray, values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    indices = np.searchsorted(times, grid, side="right") - 1
    indices = np.clip(indices, 0, len(times) - 1)
    return values[indices]


def normalized_corr(left: np.ndarray, right: np.ndarray) -> float:
    left = left - left.mean()
    right = right - right.mean()
    denom = np.linalg.norm(left) * np.linalg.norm(right)
    return float(np.dot(left, right) / denom) if denom > 0 else float("nan")


def estimate(raw_episode: Path, sample_hz: float, max_lag_ms: float) -> dict:
    streams = raw_episode / "streams"
    commands = load_jsonl(streams / "lowcmd.jsonl")
    states = load_jsonl(streams / "lowstate.jsonl")
    commands = [row for row in commands if row.get("publish_wall_ns") and row.get("q_commanded")]
    states = [row for row in states if row.get("host_receive_wall_ns") and row.get("q")]
    if len(commands) < 5 or len(states) < 5:
        raise ValueError("At least five LowCmd and LowState samples are required")

    command_t = np.asarray([row["publish_wall_ns"] for row in commands], dtype=np.float64) / 1e9
    state_t = np.asarray([row["host_receive_wall_ns"] for row in states], dtype=np.float64) / 1e9
    command_q = np.asarray([row["q_commanded"] for row in commands], dtype=np.float64)
    state_q = np.asarray([row["q"] for row in states], dtype=np.float64)
    if command_q.ndim != 2 or state_q.ndim != 2 or command_q.shape[1] != state_q.shape[1]:
        raise ValueError(
            f"LowCmd/LowState joint dimensions differ: {command_q.shape} vs {state_q.shape}"
        )
    start = max(command_t[0], state_t[0])
    stop = min(command_t[-1], state_t[-1])
    dt = 1.0 / sample_hz
    grid = np.arange(start, stop, dt)
    if len(grid) < 20:
        raise ValueError("The overlapping stream duration is too short")

    command_grid = zero_order_hold(command_t, command_q, grid)
    state_grid = interpolate_rows(state_t, state_q, grid)
    command_velocity = np.gradient(command_grid, dt, axis=0)
    state_velocity = np.gradient(state_grid, dt, axis=0)
    max_lag = min(
        int(round(max_lag_ms / 1000.0 * sample_hz)),
        len(grid) - 2,
    )
    per_joint = []
    for joint in range(command_q.shape[1]):
        command_signal = command_velocity[:, joint]
        state_signal = state_velocity[:, joint]
        if np.std(command_signal) < 1e-4 or np.std(state_signal) < 1e-4:
            per_joint.append({"joint": joint, "delay_ms": None, "correlation": None,
                              "reason": "insufficient excitation"})
            continue
        scores = []
        for lag in range(max_lag + 1):
            if lag == 0:
                score = normalized_corr(command_signal, state_signal)
            else:
                score = normalized_corr(command_signal[:-lag], state_signal[lag:])
            scores.append(score)
        finite_scores = np.isfinite(scores)
        if not np.any(finite_scores):
            per_joint.append({"joint": joint, "delay_ms": None, "correlation": None,
                              "reason": "correlation undefined"})
            continue
        best_lag = int(np.nanargmax(scores))
        per_joint.append({
            "joint": joint,
            "delay_ms": best_lag / sample_hz * 1000.0,
            "correlation": scores[best_lag],
            "reason": None,
        })

    valid_delays = [item["delay_ms"] for item in per_joint if item["delay_ms"] is not None]
    return {
        "raw_episode": str(raw_episode),
        "method": "velocity_cross_correlation",
        "sample_hz": sample_hz,
        "max_lag_ms": max_lag_ms,
        "per_joint": per_joint,
        "median_delay_ms": float(np.median(valid_delays)) if valid_delays else None,
        "note": "This estimates response onset/phase delay, not target settling time.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-episode", type=Path, required=True)
    parser.add_argument("--sample-hz", type=float, default=200.0)
    parser.add_argument("--max-lag-ms", type=float, default=1000.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = estimate(args.input_episode.expanduser().resolve(), args.sample_hz, args.max_lag_ms)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
