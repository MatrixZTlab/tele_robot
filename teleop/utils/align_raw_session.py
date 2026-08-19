#!/usr/bin/env python3
"""Align asynchronous tele_robot raw streams into Unitree-style 20 Hz episodes."""

from __future__ import annotations

import argparse
import bisect
import json
import math
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def robust_affine_clock(source: np.ndarray, host_ns: np.ndarray) -> tuple[float, float, float]:
    """Fit host_ns = scale * source + offset and return residual p95 in ms."""
    source = np.asarray(source, dtype=np.float64)
    host_ns = np.asarray(host_ns, dtype=np.float64)
    source_origin = source[0]
    host_origin = host_ns[0]
    x = source - source_origin
    y = host_ns - host_origin
    keep = np.ones(len(x), dtype=bool)
    scale = 1.0
    relative_offset = 0.0
    for _ in range(4):
        if keep.sum() < 3:
            break
        scale, relative_offset = np.polyfit(x[keep], y[keep], 1)
        residual = y - (scale * x + relative_offset)
        median = np.median(residual[keep])
        mad = np.median(np.abs(residual[keep] - median))
        if mad <= 0:
            break
        keep = np.abs(residual - median) <= 4.0 * 1.4826 * mad
    offset = host_origin + relative_offset - scale * source_origin
    residual_ms = np.abs(host_ns - (scale * source + offset)) / 1e6
    return float(scale), float(offset), float(np.percentile(residual_ms, 95))


def deduplicate_camera_events(events: list[dict[str, Any]]) -> dict[str, int]:
    """Remove repeated camera frames before fitting and timestamp matching."""
    input_events = len(events)
    deduplicated = []
    seen_sequences = set()
    seen_device_timestamps = set()
    duplicate_sequences = 0
    duplicate_device_timestamps = 0

    for event in events:
        sequence = event.get("sequence")
        device_timestamp = event.get("device_timestamp_raw")
        if sequence is not None and sequence in seen_sequences:
            duplicate_sequences += 1
            continue
        if device_timestamp is not None and device_timestamp in seen_device_timestamps:
            duplicate_device_timestamps += 1
            continue
        if sequence is not None:
            seen_sequences.add(sequence)
        if device_timestamp is not None:
            seen_device_timestamps.add(device_timestamp)
        deduplicated.append(event)

    events[:] = deduplicated
    return {
        "input_events": input_events,
        "deduplicated_events": len(deduplicated),
        "dropped_duplicate_sequences": duplicate_sequences,
        "dropped_duplicate_device_timestamps": duplicate_device_timestamps,
    }


def assign_camera_timestamps(events: list[dict[str, Any]], latency_ms: float) -> dict[str, Any]:
    deduplication = deduplicate_camera_events(events)
    usable = [
        event for event in events
        if event.get("device_timestamp_raw") is not None
        and event.get("capture_host_receive_wall_ns") is not None
    ]
    model = {
        "kind": "host_receive",
        "latency_ms": float(latency_ms),
        **deduplication,
    }
    if len(usable) >= 10:
        source = np.asarray([event["device_timestamp_raw"] for event in usable], dtype=np.float64)
        host = np.asarray([event["capture_host_receive_wall_ns"] for event in usable], dtype=np.float64)
        model["source_timestamp_backwards"] = int(np.sum(np.diff(source) < 0))
        if np.all(np.diff(source) > 0):
            scale, offset, residual_p95_ms = robust_affine_clock(source, host)
            model.update({
                "kind": "affine_device_to_capture_host",
                "scale_ns_per_raw_unit": scale,
                "offset_ns": offset,
                "residual_p95_ms": residual_p95_ms,
            })
            for event in events:
                raw = event.get("device_timestamp_raw")
                if raw is not None:
                    event["aligned_wall_ns"] = int(scale * float(raw) + offset - latency_ms * 1e6)

    for event in events:
        if "aligned_wall_ns" not in event:
            timestamp = event.get("capture_host_receive_wall_ns", event.get("timestamp_ns"))
            if timestamp is not None:
                event["aligned_wall_ns"] = int(timestamp - latency_ms * 1e6)
    events.sort(key=lambda item: item.get("aligned_wall_ns", 0))
    return model


def assign_pico_timestamps(events: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [
        event for event in events
        if event.get("source_timestamp_raw") is not None
        and event.get("host_receive_wall_ns") is not None
    ]
    model = {"kind": "host_receive"}
    if len(usable) >= 20:
        source = np.asarray([event["source_timestamp_raw"] for event in usable], dtype=np.float64)
        host = np.asarray([event["host_receive_wall_ns"] for event in usable], dtype=np.float64)
        if np.all(np.diff(source) > 0):
            scale, offset, residual_p95_ms = robust_affine_clock(source, host)
            model = {
                "kind": "affine_pico_to_control_host",
                "scale_ns_per_raw_unit": scale,
                "offset_ns": offset,
                "residual_p95_ms": residual_p95_ms,
            }
            for event in events:
                raw = event.get("source_timestamp_raw")
                if raw is not None:
                    event["aligned_wall_ns"] = int(scale * float(raw) + offset)
    for event in events:
        if "aligned_wall_ns" not in event and event.get("host_receive_wall_ns") is not None:
            event["aligned_wall_ns"] = int(event["host_receive_wall_ns"])
    events.sort(key=lambda item: item.get("aligned_wall_ns", 0))
    return model


def event_time(event: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = event.get(key)
        if value:
            return int(value)
    return None


def nearest_event(events: list[dict[str, Any]], times: list[int], target_ns: int):
    if not events:
        return None, None
    pos = bisect.bisect_left(times, target_ns)
    candidates = [index for index in (pos - 1, pos) if 0 <= index < len(events)]
    index = min(candidates, key=lambda item: abs(times[item] - target_ns))
    return events[index], times[index] - target_ns


def fixed_grid_matches(
    events: list[dict[str, Any]], times: list[int], fps: float
) -> list[dict[str, Any]]:
    """Match a true fixed-rate grid to a reference stream without changing grid time.

    The previous implementation used the matched camera timestamp as the target
    timestamp.  That inherited camera jitter and silently shortened the episode
    whenever an invalid frame was skipped.  Grid timestamps are now authoritative;
    the camera timestamp is only a matched observation with a measured error.
    """
    if not times:
        return []
    period_ns = int(round(1e9 / fps))
    targets = range(int(times[0]), int(times[-1]) + 1, period_ns)
    matches = []
    previous_identity = None
    reuse_run_length = 0
    for grid_index, target_ns in enumerate(targets):
        event, delta_ns = nearest_event(events, times, target_ns)
        identity = None
        if event is not None:
            candidate_identity = (
                event.get("stream_session_id"),
                event.get("sequence"),
                event.get("device_timestamp_raw"),
            )
            if any(value is not None for value in candidate_identity):
                identity = candidate_identity
        reused = identity is not None and identity == previous_identity
        reuse_run_length = reuse_run_length + 1 if reused else 0
        matches.append(
            {
                "grid_index": grid_index,
                "target_ns": int(target_ns),
                "event": event,
                "delta_ns": delta_ns,
                "reference_reused": reused,
                "reference_reuse_run_length": reuse_run_length,
            }
        )
        previous_identity = identity
    return matches


def select_action_event(
    events: list[dict[str, Any]],
    times: list[int],
    target_ns: int,
    mode: str,
):
    """Select the command associated with an observation target timestamp."""
    if not events:
        return None, None
    if mode == "nearest":
        return nearest_event(events, times, target_ns)
    if mode == "first-after":
        pos = bisect.bisect_left(times, target_ns)
        if pos >= len(events):
            return None, None
        return events[pos], times[pos] - target_ns
    if mode == "latest-before":
        pos = bisect.bisect_right(times, target_ns) - 1
        if pos < 0:
            return None, None
        return events[pos], times[pos] - target_ns
    raise ValueError(f"Unsupported action match mode: {mode}")


def invalid_runs(valid_grid_indices: list[int], candidate_count: int) -> list[dict[str, int | bool]]:
    """Return contiguous invalid-grid runs and whether each touches an edge."""
    valid = set(valid_grid_indices)
    runs = []
    start = None
    for index in range(candidate_count + 1):
        is_invalid = index < candidate_count and index not in valid
        if is_invalid and start is None:
            start = index
        elif not is_invalid and start is not None:
            end = index - 1
            runs.append(
                {
                    "start_grid_index": start,
                    "end_grid_index": end,
                    "length": end - start + 1,
                    "touches_edge": start == 0 or end == candidate_count - 1,
                }
            )
            start = None
    return runs


def bracket_events(events: list[dict[str, Any]], times: list[int], target_ns: int):
    pos = bisect.bisect_left(times, target_ns)
    if pos < len(events) and times[pos] == target_ns:
        event = events[pos]
        return event, event, 0.0, 0
    if pos == 0 or pos >= len(events):
        return None
    left = events[pos - 1]
    right = events[pos]
    left_t = times[pos - 1]
    right_t = times[pos]
    if right_t <= left_t:
        return None
    alpha = (target_ns - left_t) / (right_t - left_t)
    return left, right, float(alpha), right_t - left_t


def lerp_vector(left, right, alpha: float) -> list[float]:
    left_arr = np.asarray(left, dtype=np.float64)
    right_arr = np.asarray(right, dtype=np.float64)
    return ((1.0 - alpha) * left_arr + alpha * right_arr).tolist()


def matrix_to_quaternion(matrix: np.ndarray) -> np.ndarray:
    rotation = matrix[:3, :3]
    trace = np.trace(rotation)
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2
        quat = np.array([
            (rotation[2, 1] - rotation[1, 2]) / s,
            (rotation[0, 2] - rotation[2, 0]) / s,
            (rotation[1, 0] - rotation[0, 1]) / s,
            0.25 * s,
        ])
    else:
        index = int(np.argmax(np.diag(rotation)))
        if index == 0:
            s = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2
            quat = np.array([0.25 * s, (rotation[0, 1] + rotation[1, 0]) / s,
                             (rotation[0, 2] + rotation[2, 0]) / s,
                             (rotation[2, 1] - rotation[1, 2]) / s])
        elif index == 1:
            s = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2
            quat = np.array([(rotation[0, 1] + rotation[1, 0]) / s, 0.25 * s,
                             (rotation[1, 2] + rotation[2, 1]) / s,
                             (rotation[0, 2] - rotation[2, 0]) / s])
        else:
            s = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2
            quat = np.array([(rotation[0, 2] + rotation[2, 0]) / s,
                             (rotation[1, 2] + rotation[2, 1]) / s, 0.25 * s,
                             (rotation[1, 0] - rotation[0, 1]) / s])
    return quat / np.linalg.norm(quat)


def quaternion_to_matrix(quat: np.ndarray) -> np.ndarray:
    x, y, z, w = quat / np.linalg.norm(quat)
    return np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ])


def slerp(left: np.ndarray, right: np.ndarray, alpha: float) -> np.ndarray:
    dot = float(np.dot(left, right))
    if dot < 0:
        right = -right
        dot = -dot
    if dot > 0.9995:
        result = left + alpha * (right - left)
        return result / np.linalg.norm(result)
    theta = math.acos(np.clip(dot, -1.0, 1.0))
    return (math.sin((1 - alpha) * theta) * left + math.sin(alpha * theta) * right) / math.sin(theta)


def interpolate_pose(left_pose, right_pose, alpha: float) -> list[list[float]]:
    left = np.asarray(left_pose, dtype=np.float64).reshape(4, 4, order="F")
    right = np.asarray(right_pose, dtype=np.float64).reshape(4, 4, order="F")
    result = np.eye(4)
    result[:3, 3] = (1 - alpha) * left[:3, 3] + alpha * right[:3, 3]
    result[:3, :3] = quaternion_to_matrix(
        slerp(matrix_to_quaternion(left), matrix_to_quaternion(right), alpha)
    )
    return result.tolist()


def parse_latency(values: list[str]) -> dict[str, float]:
    result = {}
    for value in values:
        name, latency = value.split("=", 1)
        result[name] = float(latency)
    return result


def percentile_summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"p50": None, "p95": None, "max": None}
    data = np.abs(np.asarray(values, dtype=np.float64))
    return {
        "p50": float(np.percentile(data, 50)),
        "p95": float(np.percentile(data, 95)),
        "max": float(np.max(data)),
    }


def remove_unreferenced_files(directory: Path, referenced: set[str]) -> None:
    for path in directory.iterdir():
        if path.is_file() and str(path.relative_to(directory.parent)) not in referenced:
            path.unlink()


def sequence_summary(events: list[dict[str, Any]]) -> dict[str, int | float | None]:
    sequences = [int(event["sequence"]) for event in events if event.get("sequence") is not None]
    missing = 0
    duplicates = 0
    backwards = 0
    for previous, current in zip(sequences, sequences[1:]):
        if current > previous:
            missing += max(0, current - previous - 1)
        elif current == previous:
            duplicates += 1
        else:
            backwards += 1
    denominator = len(sequences) + missing
    return {
        "events": len(events),
        "with_sequence": len(sequences),
        "first": sequences[0] if sequences else None,
        "last": sequences[-1] if sequences else None,
        "missing": missing,
        "duplicates": duplicates,
        "backwards": backwards,
        "missing_ratio": (missing / denominator) if denominator else 0.0,
    }


def rgbd_delta_ms(event: dict[str, Any], clock_model: dict[str, Any]) -> float | None:
    color_raw = event.get("device_timestamp_raw")
    depth_raw = event.get("depth_device_timestamp_raw")
    scale = clock_model.get("scale_ns_per_raw_unit")
    if color_raw is None or depth_raw is None or scale is None:
        return None
    return (float(depth_raw) - float(color_raw)) * float(scale) / 1e6


def align_episode(args: argparse.Namespace) -> Path:
    raw_dir = args.input_episode.expanduser().resolve()
    no_depth = bool(getattr(args, "no_depth", False))
    require_pico = bool(getattr(args, "require_pico", False))
    action_offset_ms = float(getattr(args, "action_offset_ms", 0.0))
    action_match = str(getattr(args, "action_match", "latest-before"))
    gap_policy = str(getattr(args, "gap_policy", "trim-edges"))
    max_camera_hold_ms = float(getattr(args, "max_camera_hold_ms", 75.0))
    max_camera_hold_frames = int(getattr(args, "max_camera_hold_frames", 2))
    max_state_interpolation_gap_ms = float(
        getattr(args, "max_state_interpolation_gap_ms", 150.0)
    )
    max_action_hold_ms = float(getattr(args, "max_action_hold_ms", 150.0))
    max_imputed_ratio = float(getattr(args, "max_imputed_ratio", 0.10))
    max_sequence_missing_ratio = float(
        getattr(args, "max_sequence_missing_ratio", 0.05)
    )
    manifest_path = raw_dir / "manifest.json"
    with manifest_path.open("r", encoding="utf-8") as handle:
        raw_manifest = json.load(handle)
    streams_dir = raw_dir / "streams"
    camera_files = sorted(streams_dir.glob("camera_*.jsonl"))
    if not camera_files:
        raise FileNotFoundError(f"No camera streams found in {streams_dir}")

    camera_latency = parse_latency(args.camera_latency_ms)
    cameras = {}
    camera_models = {}
    camera_sequence_quality = {}
    for path in camera_files:
        name = path.stem.removeprefix("camera_")
        cameras[name] = load_jsonl(path)
        camera_sequence_quality[name] = sequence_summary(cameras[name])
        camera_models[name] = assign_camera_timestamps(
            cameras[name], camera_latency.get(name, 0.0)
        )

    reference = args.reference_camera or next(iter(cameras))
    if reference not in cameras:
        raise KeyError(f"Reference camera {reference!r} not found; choices: {sorted(cameras)}")

    lowstate = load_jsonl(streams_dir / "lowstate.jsonl")
    lowcmd = load_jsonl(streams_dir / "lowcmd.jsonl")
    pico = load_jsonl(streams_dir / "pico.jsonl")
    pico_model = assign_pico_timestamps(pico)

    clock_issues = []
    for name, model in camera_models.items():
        if model.get("kind") != "affine_device_to_capture_host":
            clock_issues.append(f"camera:{name}:device_clock_unavailable")
        elif model.get("residual_p95_ms", float("inf")) > args.max_clock_fit_residual_ms:
            clock_issues.append(f"camera:{name}:clock_fit_residual")
        if not no_depth and any(event.get("depth_path") for event in cameras[name]) and not any(
            event.get("device_timestamp_raw") is not None
            and event.get("depth_device_timestamp_raw") is not None
            for event in cameras[name]
        ):
            clock_issues.append(f"camera:{name}:depth_clock_unavailable")
    if require_pico:
        if pico_model.get("kind") != "affine_pico_to_control_host":
            clock_issues.append("pico:source_clock_unavailable")
        elif pico_model.get("residual_p95_ms", float("inf")) > args.max_clock_fit_residual_ms:
            clock_issues.append("pico:clock_fit_residual")
    control_clock = raw_manifest.get("control_host_clock_sync")
    if control_clock is not None and not control_clock.get("valid", False):
        clock_issues.append("control_host:chrony_invalid")
    if args.require_source_clocks and clock_issues:
        raise RuntimeError(
            "Strict source-clock alignment failed: " + ", ".join(clock_issues)
        )

    for event in lowstate:
        host_time = event_time(event, "host_receive_wall_ns")
        source_time = event_time(event, "source_timestamp_ns")
        # Only use a robot-provided source stamp when it is demonstrably in the
        # synchronized wall-clock domain. Robot uptime/tick counters must not be
        # compared directly with Unix time.
        event["aligned_wall_ns"] = (
            source_time
            if source_time and host_time and abs(source_time - host_time) < 60_000_000_000
            else host_time
        )
    for event in lowcmd:
        event["aligned_wall_ns"] = event_time(event, "publish_wall_ns", "target_update_wall_ns")
    lowstate = sorted((e for e in lowstate if e["aligned_wall_ns"]), key=lambda e: e["aligned_wall_ns"])
    lowcmd = sorted((e for e in lowcmd if e["aligned_wall_ns"]), key=lambda e: e["aligned_wall_ns"])
    pico = [e for e in pico if e.get("aligned_wall_ns")]

    camera_times = {name: [int(e["aligned_wall_ns"]) for e in events] for name, events in cameras.items()}
    state_times = [int(e["aligned_wall_ns"]) for e in lowstate]
    command_times = [int(e["aligned_wall_ns"]) for e in lowcmd]
    pico_times = [int(e["aligned_wall_ns"]) for e in pico]
    reference_times = camera_times[reference]
    if not reference_times or not lowstate or not lowcmd:
        raise ValueError("Alignment requires reference camera, LowState, and LowCmd streams")

    grid_matches = fixed_grid_matches(cameras[reference], reference_times, args.fps)

    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else raw_dir.parent.parent / "aligned" / raw_dir.name
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists; use --overwrite: {output_dir}")
        shutil.rmtree(output_dir)
    colors_dir = output_dir / "colors"
    depths_dir = output_dir / "depths"
    colors_dir.mkdir(parents=True)
    depths_dir.mkdir(parents=True)

    frames = []
    invalid_counts: dict[str, int] = {}
    camera_errors_ms = {name: [] for name in cameras}
    rgbd_errors_ms = {name: [] for name in cameras}
    state_gaps_ms = []
    pico_gaps_ms = []
    command_ages_ms = []
    imputation_counts: dict[str, int] = {}
    camera_previous_identity = {name: None for name in cameras}
    camera_reuse_run_length = {name: 0 for name in cameras}

    def invalidate(reasons, reason):
        reasons.append(reason)
        invalid_counts[reason] = invalid_counts.get(reason, 0) + 1

    def impute(reasons, reason):
        reasons.append(reason)
        imputation_counts[reason] = imputation_counts.get(reason, 0) + 1

    valid_grid_indices = []
    for grid_match in grid_matches:
        grid_index = int(grid_match["grid_index"])
        target_ns = int(grid_match["target_ns"])
        reasons = []
        imputation_reasons = []
        colors = {}
        depths = {}
        camera_alignment = {}
        for camera_index, (name, events) in enumerate(cameras.items()):
            event, delta_ns = nearest_event(events, camera_times[name], target_ns)
            delta_ms = delta_ns / 1e6 if delta_ns is not None else None
            camera_alignment[name] = delta_ms
            identity = None
            if event is not None:
                identity = (
                    event.get("stream_session_id"),
                    event.get("sequence"),
                    event.get("device_timestamp_raw"),
                )
                if not any(value is not None for value in identity):
                    identity = None
            reused = identity is not None and identity == camera_previous_identity[name]
            camera_reuse_run_length[name] = (
                camera_reuse_run_length[name] + 1 if reused else 0
            )
            camera_previous_identity[name] = identity
            if delta_ms is None or abs(delta_ms) > max_camera_hold_ms:
                invalidate(reasons, f"camera:{name}:unmatched")
                continue
            if camera_reuse_run_length[name] > max_camera_hold_frames:
                invalidate(reasons, f"camera:{name}:hold_too_long")
                continue
            if abs(delta_ms) > args.max_camera_error_ms:
                impute(imputation_reasons, f"camera:{name}:nearest_hold")
            if reused:
                impute(imputation_reasons, f"camera:{name}:reused_for_grid")
            camera_errors_ms[name].append(delta_ms)
            if not no_depth:
                depth_delta_ms = rgbd_delta_ms(event, camera_models[name])
                if depth_delta_ms is not None:
                    rgbd_errors_ms[name].append(depth_delta_ms)
                    if abs(depth_delta_ms) > args.max_rgbd_error_ms:
                        invalidate(reasons, f"camera:{name}:rgbd_skew")
            try:
                source_image = raw_dir / event["image_path"]
                color_name = f"{grid_index:06d}_color_{camera_index}.jpg"
                shutil.copy2(source_image, colors_dir / color_name)
                colors[f"color_{camera_index}"] = str(Path("colors") / color_name)
            except (KeyError, OSError):
                invalidate(reasons, f"camera:{name}:image_missing")
            if not no_depth and event.get("depth_path") and event.get("depth_shape"):
                try:
                    dtype = np.dtype(event.get("depth_dtype", "uint16"))
                    depth = np.fromfile(raw_dir / event["depth_path"], dtype=dtype).reshape(event["depth_shape"])
                    depth_name = f"{grid_index:06d}_depth_{camera_index}.png"
                    if not cv2.imwrite(str(depths_dir / depth_name), depth):
                        raise OSError("cv2.imwrite returned false")
                    depths[f"depth_{camera_index}"] = str(Path("depths") / depth_name)
                except (OSError, TypeError, ValueError):
                    invalidate(reasons, f"camera:{name}:depth_invalid")

        state_bracket = bracket_events(lowstate, state_times, target_ns)
        if state_bracket is None:
            invalidate(reasons, "lowstate:no_bracket")
            continue
        state_left, state_right, state_alpha, state_gap_ns = state_bracket
        state_gap_ms = state_gap_ns / 1e6
        state_gaps_ms.append(state_gap_ms)
        if state_gap_ms > max_state_interpolation_gap_ms:
            invalidate(reasons, "lowstate:gap")
        elif state_gap_ms > args.max_state_gap_ms:
            impute(imputation_reasons, "lowstate:short_interpolation")
        try:
            q = lerp_vector(state_left["q"], state_right["q"], state_alpha)
            dq = lerp_vector(state_left["dq"], state_right["dq"], state_alpha)
            if len(q) < 14 or len(dq) < 14:
                raise ValueError("LowState requires 14 arm joints")
        except (KeyError, TypeError, ValueError):
            invalidate(reasons, "lowstate:invalid")
            continue

        action_target_ns = target_ns + int(round(action_offset_ms * 1e6))
        command, command_delta_ns = select_action_event(
            lowcmd, command_times, action_target_ns, action_match
        )
        if command is None:
            invalidate(reasons, "lowcmd:missing")
            continue
        command_age_ms = abs(float(command_delta_ns)) / 1e6
        command_ages_ms.append(command_age_ms)
        if command_age_ms > max_action_hold_ms:
            invalidate(reasons, "lowcmd:stale")
        elif command_age_ms > args.max_action_age_ms:
            impute(imputation_reasons, "lowcmd:short_hold")
        action = command.get("q_commanded", command.get("q_ik_target"))
        if action is None or len(action) < 14:
            invalidate(reasons, "lowcmd:invalid")
            continue
        ee_right_left = command.get("ee_command_right_left") or []
        right_ee = [float(ee_right_left[0])] if len(ee_right_left) > 0 else []
        left_ee = [float(ee_right_left[1])] if len(ee_right_left) > 1 else []

        pico_data = None
        pico_bracket = bracket_events(pico, pico_times, target_ns)
        if pico_bracket is None:
            if require_pico:
                invalidate(reasons, "pico:no_bracket")
        else:
            pico_left, pico_right, pico_alpha, pico_gap_ns = pico_bracket
            pico_gap_ms = pico_gap_ns / 1e6
            pico_gaps_ms.append(pico_gap_ms)
            if require_pico and pico_gap_ms > args.max_pico_gap_ms:
                invalidate(reasons, "pico:gap")
            try:
                pico_data = {
                    "left_pose": interpolate_pose(pico_left["left_pose"], pico_right["left_pose"], pico_alpha),
                    "right_pose": interpolate_pose(pico_left["right_pose"], pico_right["right_pose"], pico_alpha),
                    "left_state": pico_left.get("left_state"),
                    "right_state": pico_left.get("right_state"),
                }
            except (KeyError, TypeError, ValueError, np.linalg.LinAlgError):
                if require_pico:
                    invalidate(reasons, "pico:invalid_pose")
                pico_data = None

        if reasons and not args.keep_invalid:
            continue
        frame_index = len(frames)
        if not reasons:
            valid_grid_indices.append(grid_index)
        frames.append({
            "idx": frame_index,
            "frame_index": frame_index,
            "timestamp": 0.0,
            "episode_index": int(raw_dir.name.split("_")[-1]),
            "colors": colors,
            "depths": depths,
            "states": {
                "left_arm": {"qpos": q[:7], "qvel": [], "torque": []},
                "right_arm": {"qpos": q[7:14], "qvel": [], "torque": []},
                "left_ee": {"qpos": left_ee, "qvel": [], "torque": []},
                "right_ee": {"qpos": right_ee, "qvel": [], "torque": []},
                "body": {"qpos": []},
            },
            "actions": {
                "left_arm": {"qpos": action[:7], "qvel": [], "torque": []},
                "right_arm": {"qpos": action[7:14], "qvel": [], "torque": []},
                "left_ee": {"qpos": left_ee, "qvel": [], "torque": []},
                "right_ee": {"qpos": right_ee, "qvel": [], "torque": []},
                "body": {"qpos": []},
            },
            "pico": pico_data,
            "alignment": {
                "scheme": "fixed_grid_interpolation_v2",
                "valid": not reasons,
                "invalid_reasons": reasons,
                "imputed": bool(imputation_reasons),
                "imputation_reasons": imputation_reasons,
                "grid_index": grid_index,
                "grid_target_wall_ns": target_ns,
                "reference_wall_ns": int(grid_match["event"]["aligned_wall_ns"]),
                "camera_delta_ms": camera_alignment,
                "lowstate_gap_ms": state_gap_ms,
                "pico_gap_ms": pico_gaps_ms[-1] if pico_data is not None else None,
                "lowcmd_age_ms": command_age_ms,
                "action_target_wall_ns": action_target_ns,
                "action_time_error_ms": float(command_delta_ns) / 1e6,
                "action_match": action_match,
                "action_offset_ms": action_offset_ms,
                "lowcmd_sequence": command.get("sequence"),
            },
        })

    if not frames:
        raise RuntimeError("No valid aligned frames were produced; inspect thresholds and raw streams")

    # Rebase to episode time while retaining the true fixed-grid spacing.  With
    # trim-edges this remains exactly 1/fps; with explicit legacy compression it
    # exposes any interior hole instead of disguising it as contiguous data.
    first_grid_target_ns = int(frames[0]["alignment"]["grid_target_wall_ns"])
    for frame_index, frame in enumerate(frames):
        frame["idx"] = frame_index
        frame["frame_index"] = frame_index
        frame["timestamp"] = (
            int(frame["alignment"]["grid_target_wall_ns"]) - first_grid_target_ns
        ) / 1e9

    remove_unreferenced_files(
        colors_dir,
        {
            image_path
            for frame in frames
            for image_path in frame["colors"].values()
        },
    )
    remove_unreferenced_files(
        depths_dir,
        {
            depth_path
            for frame in frames
            for depth_path in frame["depths"].values()
        },
    )

    episode = {
        "info": {
            "version": "2.0.0",
            "fps": args.fps,
            "tolerance_s": 1e-4,
            "alignment": {
                "scheme": "fixed_grid_interpolation_v2",
                "reference_camera": reference,
                "gap_policy": gap_policy,
                "pico_required": require_pico,
                "action_match": action_match,
                "action_offset_ms": action_offset_ms,
            },
        },
        "text": {
            "goal": raw_manifest.get("metadata", {}).get("task_goal", ""),
            "desc": "timestamp-aligned raw teleoperation episode",
            "steps": "",
        },
        "data": frames,
    }
    with (output_dir / "data.json").open("w", encoding="utf-8") as handle:
        json.dump(episode, handle, ensure_ascii=False, indent=2)

    valid_frame_count = len(valid_grid_indices)
    valid_ratio = valid_frame_count / len(grid_matches) if grid_matches else 0.0
    imputed_frame_count = sum(
        bool(frame.get("alignment", {}).get("imputed")) for frame in frames
    )
    imputed_ratio = imputed_frame_count / len(frames) if frames else 0.0
    timeline_invalid_runs = invalid_runs(valid_grid_indices, len(grid_matches))
    internal_invalid_runs = [run for run in timeline_invalid_runs if not run["touches_edge"]]
    sequence_quality = {
        **{
            f"camera.{name}": camera_sequence_quality[name]
            for name in cameras
        },
        "pico": sequence_summary(pico),
        "lowstate": sequence_summary(lowstate),
        "lowcmd": sequence_summary(lowcmd),
    }
    quality_issues = []
    quality_warnings = []
    if raw_manifest.get("complete") is False:
        quality_issues.append("raw_manifest_incomplete")
    if any(int(value) > 0 for value in raw_manifest.get("dropped", {}).values()):
        quality_issues.append("raw_writer_queue_drops")
    if any(int(value) > 0 for value in raw_manifest.get("failed", {}).values()):
        quality_issues.append("raw_writer_failures")
    for stream, summary in sequence_quality.items():
        if stream.startswith("camera.") or stream == "lowstate" or (
            require_pico and stream == "pico"
        ):
            if summary["backwards"]:
                quality_issues.append(f"{stream}:sequence_discontinuity")
            elif float(summary.get("missing_ratio", 0.0)) > max_sequence_missing_ratio:
                quality_issues.append(f"{stream}:excessive_sequence_loss")
            elif summary["missing"]:
                quality_warnings.append(f"{stream}:minor_sequence_loss")
    pico_producer_drops = max(
        (int(event.get("producer_drop_count", 0)) for event in pico),
        default=0,
    )
    if require_pico and pico_producer_drops:
        quality_issues.append("pico:producer_queue_drops")
    if 1.0 - valid_ratio > args.max_invalid_ratio:
        quality_issues.append("aligned_invalid_ratio")
    if imputed_ratio > max_imputed_ratio:
        quality_issues.append("aligned_imputed_ratio")
    timeline_rejected = False
    if gap_policy == "reject" and timeline_invalid_runs:
        quality_issues.append("timeline:invalid_grid_targets")
        timeline_rejected = True
    elif gap_policy == "trim-edges" and internal_invalid_runs:
        quality_issues.append("timeline:internal_gap")
        timeline_rejected = True

    report = {
        "input": str(raw_dir),
        "output": str(output_dir),
        "reference_camera": reference,
        "candidate_frames": len(grid_matches),
        "written_frames": len(frames),
        "valid_frames": valid_frame_count,
        "valid_ratio": valid_ratio,
        "invalid_counts": invalid_counts,
        "imputed_frames": imputed_frame_count,
        "imputed_ratio": imputed_ratio,
        "imputation_counts": imputation_counts,
        "raw_manifest_complete": raw_manifest.get("complete"),
        "raw_queue_dropped": raw_manifest.get("dropped", {}),
        "raw_write_failed": raw_manifest.get("failed", {}),
        "control_host_clock_sync": control_clock,
        "sequence_quality": sequence_quality,
        "pico_producer_queue_drops": pico_producer_drops,
        "quality_valid": not quality_issues,
        "quality_issues": quality_issues,
        "quality_warnings": quality_warnings,
        "timeline": {
            "scheme": "fixed_grid_interpolation_v2",
            "fps": args.fps,
            "period_ns": int(round(1e9 / args.fps)),
            "gap_policy": gap_policy,
            "invalid_runs": timeline_invalid_runs,
            "internal_invalid_runs": internal_invalid_runs,
            "rejected": timeline_rejected,
        },
        "pico_required": require_pico,
        "action_alignment": {
            "match": action_match,
            "offset_ms": action_offset_ms,
        },
        "camera_clock_models": camera_models,
        "pico_clock_model": pico_model,
        "source_clock_valid": not clock_issues,
        "source_clock_issues": clock_issues,
        "camera_abs_error_ms": {
            name: percentile_summary(values) for name, values in camera_errors_ms.items()
        },
        "rgbd_abs_error_ms": {
            name: percentile_summary(values) for name, values in rgbd_errors_ms.items()
        },
        "lowstate_gap_ms": percentile_summary(state_gaps_ms),
        "pico_gap_ms": percentile_summary(pico_gaps_ms),
        "lowcmd_age_ms": percentile_summary(command_ages_ms),
        "thresholds_ms": {
            "camera": args.max_camera_error_ms,
            "rgbd": args.max_rgbd_error_ms,
            "lowstate_gap": args.max_state_gap_ms,
            "pico_gap": args.max_pico_gap_ms,
            "action_age": args.max_action_age_ms,
            "camera_hold": max_camera_hold_ms,
            "camera_hold_frames": max_camera_hold_frames,
            "state_interpolation_gap": max_state_interpolation_gap_ms,
            "action_hold": max_action_hold_ms,
            "clock_fit_residual": args.max_clock_fit_residual_ms,
            "invalid_ratio": args.max_invalid_ratio,
            "imputed_ratio": max_imputed_ratio,
            "sequence_missing_ratio": max_sequence_missing_ratio,
        },
    }
    report_path = output_dir / "alignment_report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    if timeline_rejected:
        raise RuntimeError(
            f"Timeline policy {gap_policy!r} rejected the episode because invalid "
            f"fixed-grid targets would create an internal discontinuity. Inspect {report_path}"
        )
    if args.require_quality and quality_issues:
        raise RuntimeError(
            "Strict alignment quality check failed: "
            + ", ".join(quality_issues)
            + f". Inspect {report_path}"
        )
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-episode", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--reference-camera", default="head_camera")
    parser.add_argument("--camera-latency-ms", action="append", default=[], metavar="NAME=MS")
    parser.add_argument("--max-camera-error-ms", type=float, default=20.0)
    parser.add_argument(
        "--max-camera-hold-ms", type=float, default=75.0,
        help="Permit a nearest cached frame only within this hard age bound",
    )
    parser.add_argument(
        "--max-camera-hold-frames", type=int, default=2,
        help="Maximum consecutive fixed-grid targets allowed to reuse one camera frame",
    )
    parser.add_argument("--max-rgbd-error-ms", type=float, default=10.0)
    parser.add_argument("--no-depth", action="store_true",
                        help="Ignore raw depth streams and do not write aligned depth PNGs")
    parser.add_argument("--max-state-gap-ms", type=float, default=30.0)
    parser.add_argument(
        "--max-state-interpolation-gap-ms", type=float, default=150.0,
        help="Hard bound for interpolating a short LowState gap",
    )
    parser.add_argument("--max-pico-gap-ms", type=float, default=35.0)
    parser.add_argument("--max-action-age-ms", type=float, default=100.0)
    parser.add_argument(
        "--max-action-hold-ms", type=float, default=150.0,
        help="Hard bound for holding the most recent command",
    )
    parser.add_argument(
        "--action-offset-ms",
        type=float,
        default=0.0,
        help="Pair each observation at t with a command near t + offset (ms)",
    )
    parser.add_argument(
        "--action-match",
        choices=("latest-before", "nearest", "first-after"),
        default="latest-before",
        help="Temporal rule used to pair LowCmd with each observation",
    )
    parser.add_argument("--max-clock-fit-residual-ms", type=float, default=2.0)
    parser.add_argument("--max-invalid-ratio", type=float, default=0.05)
    parser.add_argument(
        "--max-imputed-ratio", type=float, default=0.10,
        help="Quality-report limit for frames using bounded hold/interpolation",
    )
    parser.add_argument(
        "--max-sequence-missing-ratio", type=float, default=0.05,
        help="Quality failure threshold for missing source sequence numbers",
    )
    parser.add_argument(
        "--require-pico",
        action="store_true",
        help="Require a valid Pico pose at every training frame (off by default)",
    )
    parser.add_argument("--require-source-clocks", action="store_true",
                        help="Fail unless every required source exposes a stable source clock")
    parser.add_argument("--require-quality", action="store_true",
                        help="Fail on raw drops, sequence gaps, or excessive invalid frames")
    parser.add_argument("--keep-invalid", action="store_true")
    parser.add_argument(
        "--gap-policy",
        choices=("reject", "trim-edges", "compress"),
        default="trim-edges",
        help="Never compress internal gaps unless legacy 'compress' is explicitly selected",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    print(align_episode(parse_args()))


if __name__ == "__main__":
    main()
