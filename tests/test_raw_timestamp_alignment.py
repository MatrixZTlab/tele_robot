import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from teleop.utils.align_raw_session import align_episode
from teleop.utils.analyze_actuation_delay import estimate
from teleop.utils.raw_session_writer import RawSessionWriter


def test_raw_writer_keeps_independent_streams(tmp_path):
    writer = RawSessionWriter(tmp_path / "task", queue_size=32)
    session = writer.start_episode(1, {"task_goal": "test"})
    assert writer.append_event("lowstate", {"sequence": 1, "q": [0.0] * 14})
    assert writer.append_camera_packet("head_camera", {
        "sequence": 7,
        "jpg": b"jpeg-bytes",
        "timestamp_ns": time.time_ns(),
    })
    writer.stop_episode()
    writer.close()

    assert (session / "streams" / "lowstate.jsonl").exists()
    camera_index = json.loads((session / "streams" / "camera_head_camera.jsonl").read_text())
    assert camera_index["sequence"] == 7
    assert (session / camera_index["image_path"]).read_bytes() == b"jpeg-bytes"
    manifest = json.loads((session / "manifest.json").read_text())
    assert manifest["complete"] is True
    assert manifest["written"]["camera_head_camera"] == 1


def test_raw_writer_marks_episode_incomplete_when_background_write_fails(tmp_path):
    writer = RawSessionWriter(tmp_path / "task", queue_size=8)
    session = writer.start_episode(2)

    def fail_write(*_args, **_kwargs):
        raise OSError("simulated disk failure")

    writer._write_event = fail_write
    assert writer.append_event("lowstate", {"sequence": 1})
    with pytest.raises(RuntimeError, match="simulated disk failure"):
        writer.stop_episode()
    writer.close()

    manifest = json.loads((session / "manifest.json").read_text())
    assert manifest["complete"] is False
    assert manifest["failed"]["lowstate"] == 1
    assert manifest["errors"][0]["error_type"] == "OSError"


def test_raw_writer_discards_active_episode(tmp_path):
    writer = RawSessionWriter(tmp_path / "task", queue_size=8)
    session = writer.start_episode(3)
    assert writer.append_event("lowstate", {"sequence": 1})

    discarded = writer.discard_episode(3)

    assert discarded == session
    assert not session.exists()
    assert writer.active is False
    writer.close()


def _write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _pose(x):
    matrix = np.eye(4)
    matrix[0, 3] = x
    return matrix.reshape(-1, order="F").tolist()


def test_offline_aligner_matches_cameras_and_interpolates_state(tmp_path):
    raw = tmp_path / "task" / "raw" / "episode_0001"
    streams = raw / "streams"
    cameras = raw / "cameras"
    base = 1_800_000_000_000_000_000
    duration_s = 1.0

    manifest = {
        "format": "tele_robot_raw_v1",
        "episode_index": 1,
        "metadata": {"task_goal": "move"},
    }
    raw.mkdir(parents=True)
    (raw / "manifest.json").write_text(json.dumps(manifest))

    for camera_name, phase_ms in (("head_camera", 0.0), ("left_wrist_camera", 4.0)):
        rows = []
        camera_dir = cameras / camera_name
        camera_dir.mkdir(parents=True)
        for sequence in range(31):
            t_ns = int(sequence / 30.0 * 1e9 + phase_ms * 1e6)
            image = np.full((4, 6, 3), sequence, dtype=np.uint8)
            ok, encoded = cv2.imencode(".jpg", image)
            assert ok
            image_path = camera_dir / f"{sequence:09d}.jpg"
            image_path.write_bytes(encoded.tobytes())
            row = {
                "sequence": sequence,
                "device_timestamp_raw": t_ns / 1e6,
                "capture_host_receive_wall_ns": base + t_ns + 2_000_000,
                "image_path": str(image_path.relative_to(raw)),
            }
            rows.append(row)
            if sequence in (8, 16):
                rows.append(dict(row))
        _write_jsonl(streams / f"camera_{camera_name}.jsonl", rows)

    state_rows = []
    for sequence in range(101):
        t_ns = int(sequence / 100.0 * 1e9)
        q = [sequence / 100.0] * 14
        state_rows.append({
            "sequence": sequence,
            "host_receive_wall_ns": base + t_ns,
            "q": q,
            "dq": [1.0] * 14,
        })
    _write_jsonl(streams / "lowstate.jsonl", state_rows)

    command_rows = []
    for sequence in range(21):
        t_ns = int(sequence / 20.0 * 1e9)
        command_rows.append({
            "sequence": sequence,
            "publish_wall_ns": base + t_ns,
            "q_commanded": [sequence / 20.0] * 14,
        })
    _write_jsonl(streams / "lowcmd.jsonl", command_rows)

    pico_rows = []
    for sequence in range(61):
        t_ns = int(sequence / 60.0 * 1e9)
        pico_rows.append({
            "sequence": sequence,
            "source_timestamp_raw": t_ns / 1e6,
            "host_receive_wall_ns": base + t_ns + 3_000_000,
            "left_pose": _pose(sequence / 60.0),
            "right_pose": _pose(sequence / 60.0),
            "left_state": {},
            "right_state": {},
        })
    _write_jsonl(streams / "pico.jsonl", pico_rows)

    output = tmp_path / "aligned" / "episode_0001"
    args = argparse.Namespace(
        input_episode=raw,
        output_dir=output,
        fps=20.0,
        reference_camera="head_camera",
        camera_latency_ms=[],
        max_camera_error_ms=20.0,
        max_rgbd_error_ms=10.0,
        max_state_gap_ms=30.0,
        max_pico_gap_ms=35.0,
        max_action_age_ms=100.0,
        max_clock_fit_residual_ms=2.0,
        max_invalid_ratio=0.10,
        require_source_clocks=True,
        require_quality=True,
        keep_invalid=False,
        overwrite=False,
    )
    assert align_episode(args) == output

    episode = json.loads((output / "data.json").read_text())
    assert len(episode["data"]) >= 18
    frame = episode["data"][5]
    assert frame["alignment"]["valid"] is True
    assert abs(frame["alignment"]["camera_delta_ms"]["left_wrist_camera"]) <= 20.0
    assert len(frame["states"]["left_arm"]["qpos"]) == 7
    assert frame["states"]["left_arm"]["qvel"] == []
    assert frame["states"]["right_arm"]["qvel"] == []
    assert frame["states"]["left_ee"]["qpos"] == []
    assert frame["states"]["right_ee"]["qpos"] == []
    assert len(frame["actions"]["right_arm"]["qpos"]) == 7
    report = json.loads((output / "alignment_report.json").read_text())
    assert report["camera_clock_models"]["head_camera"]["kind"] == "affine_device_to_capture_host"
    assert report["camera_clock_models"]["head_camera"]["dropped_duplicate_sequences"] == 2
    assert report["sequence_quality"]["camera.head_camera"]["duplicates"] == 2
    assert report["valid_frames"] == len(episode["data"])
    assert report["raw_queue_dropped"] == {}
    referenced_images = {
        image_path
        for item in episode["data"]
        for image_path in item["colors"].values()
    }
    written_images = {
        str(path.relative_to(output))
        for path in (output / "colors").iterdir()
    }
    assert written_images == referenced_images


def test_actuation_delay_estimator_recovers_known_lag(tmp_path):
    raw = tmp_path / "episode_0002"
    streams = raw / "streams"
    base = 1_800_000_000_000_000_000
    delay_s = 0.08
    commands = []
    states = []
    for sequence in range(501):
        t = sequence / 100.0
        command_value = np.sin(2 * np.pi * 0.7 * t)
        state_value = np.sin(2 * np.pi * 0.7 * (t - delay_s))
        commands.append({
            "sequence": sequence,
            "publish_wall_ns": base + int(t * 1e9),
            "q_commanded": [command_value] * 14,
        })
        states.append({
            "sequence": sequence,
            "host_receive_wall_ns": base + int(t * 1e9),
            "q": [state_value] * 14,
        })
    _write_jsonl(streams / "lowcmd.jsonl", commands)
    _write_jsonl(streams / "lowstate.jsonl", states)

    report = estimate(raw, sample_hz=100.0, max_lag_ms=300.0)
    assert abs(report["median_delay_ms"] - 80.0) <= 20.0
