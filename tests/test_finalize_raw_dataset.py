from argparse import Namespace
from pathlib import Path

from teleop.utils.finalize_raw_dataset import (
    _alignment_args,
    build_quality_summary,
)


def test_practical_alignment_profile_is_20_hz_and_offline():
    args = Namespace(
        fps=20.0,
        reference_camera="head_camera",
        max_camera_error_ms=20.0,
        max_state_gap_ms=60.0,
        max_pico_gap_ms=50.0,
        max_action_age_ms=100.0,
        action_offset_ms=25.0,
        action_match="nearest",
        max_clock_fit_residual_ms=20.0,
        max_invalid_ratio=0.05,
        require_source_clocks=False,
        require_pico=False,
        require_quality=False,
        gap_policy="trim-edges",
        overwrite_aligned=False,
    )
    result = _alignment_args(args, Path("raw/episode_0001"), Path("aligned/episode_0001"))

    assert result.fps == 20.0
    assert result.max_state_gap_ms == 60.0
    assert result.max_state_interpolation_gap_ms == 150.0
    assert result.max_camera_hold_ms == 75.0
    assert result.max_camera_hold_frames == 2
    assert result.max_action_hold_ms == 150.0
    assert result.max_imputed_ratio == 0.10
    assert result.max_sequence_missing_ratio == 0.05
    assert result.max_pico_gap_ms == 50.0
    assert result.keep_invalid is False
    assert result.require_pico is False
    assert result.gap_policy == "trim-edges"
    assert result.action_offset_ms == 25.0
    assert result.action_match == "nearest"


def test_quality_summary_uses_candidate_grain():
    report = {
        "input": "/task/raw/episode_0001",
        "candidate_frames": 100,
        "valid_frames": 96,
        "valid_ratio": 0.96,
        "imputed_frames": 3,
        "imputed_ratio": 0.03125,
        "imputation_counts": {"camera:left_wrist_camera:nearest_hold": 3},
        "quality_valid": True,
        "quality_issues": [],
        "source_clock_valid": True,
        "source_clock_issues": [],
        "invalid_counts": {"lowstate:gap": 4},
        "camera_abs_error_ms": {"head_camera": {"p95": 0.0}},
        "lowstate_gap_ms": {"p95": 42.0},
        "pico_gap_ms": {"p95": 35.0},
        "lowcmd_age_ms": {"p95": 49.0},
        "raw_queue_dropped": {},
        "raw_write_failed": {},
    }

    summary = build_quality_summary(Path("/task"), [report], [])

    assert summary["candidate_frames"] == 100
    assert summary["valid_frames"] == 96
    assert summary["valid_ratio"] == 0.96
    assert summary["all_quality_valid"] is True
    assert summary["episodes"][0]["episode"] == "episode_0001"
    assert summary["episodes"][0]["imputed_frames"] == 3
    assert summary["episodes"][0]["imputed_ratio"] == 0.03125
