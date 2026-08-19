#!/usr/bin/env python3
"""Batch-align raw teleoperation episodes, report quality, and optionally export LeRobot."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from teleop.utils.align_raw_session import align_episode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task-dir",
        type=Path,
        required=True,
        help="task directory containing raw/episode_XXXX",
    )
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--reference-camera", default="head_camera")
    parser.add_argument("--max-camera-error-ms", type=float, default=20.0)
    parser.add_argument("--max-camera-hold-ms", type=float, default=75.0)
    parser.add_argument("--max-camera-hold-frames", type=int, default=2)
    parser.add_argument("--max-state-gap-ms", type=float, default=60.0)
    parser.add_argument("--max-state-interpolation-gap-ms", type=float, default=150.0)
    parser.add_argument("--max-pico-gap-ms", type=float, default=50.0)
    parser.add_argument("--max-action-age-ms", type=float, default=100.0)
    parser.add_argument("--max-action-hold-ms", type=float, default=150.0)
    parser.add_argument("--action-offset-ms", type=float, default=0.0)
    parser.add_argument(
        "--action-match",
        choices=("latest-before", "nearest", "first-after"),
        default="latest-before",
    )
    parser.add_argument("--max-clock-fit-residual-ms", type=float, default=20.0)
    parser.add_argument("--max-invalid-ratio", type=float, default=0.05)
    parser.add_argument("--max-imputed-ratio", type=float, default=0.10)
    parser.add_argument("--max-sequence-missing-ratio", type=float, default=0.05)
    parser.add_argument(
        "--no-depth",
        action="store_true",
        help="Align and export RGB only; raw depth files remain untouched",
    )
    parser.add_argument("--require-source-clocks", action="store_true")
    parser.add_argument(
        "--require-pico",
        action="store_true",
        help="Make Pico pose availability part of the training-frame quality gate",
    )
    parser.add_argument("--require-quality", action="store_true")
    parser.add_argument(
        "--gap-policy",
        choices=("reject", "trim-edges", "compress"),
        default="trim-edges",
        help="Default refuses internal timeline gaps; compress is legacy-only",
    )
    parser.add_argument("--overwrite-aligned", action="store_true")
    parser.add_argument(
        "--export-lerobot",
        action="store_true",
        help="export aligned episodes after every alignment succeeds",
    )
    parser.add_argument("--lerobot-output", type=Path)
    parser.add_argument("--repo-id")
    parser.add_argument("--video-codec", default="h264")
    parser.add_argument("--overwrite-lerobot", action="store_true")
    return parser.parse_args()


def _alignment_args(
    args: argparse.Namespace, raw_episode: Path, output_episode: Path
) -> SimpleNamespace:
    return SimpleNamespace(
        input_episode=raw_episode,
        output_dir=output_episode,
        fps=args.fps,
        reference_camera=args.reference_camera,
        camera_latency_ms=[],
        max_camera_error_ms=args.max_camera_error_ms,
        max_camera_hold_ms=getattr(args, "max_camera_hold_ms", 75.0),
        max_camera_hold_frames=getattr(args, "max_camera_hold_frames", 2),
        max_rgbd_error_ms=10.0,
        no_depth=bool(getattr(args, "no_depth", False)),
        max_state_gap_ms=args.max_state_gap_ms,
        max_state_interpolation_gap_ms=getattr(
            args, "max_state_interpolation_gap_ms", 150.0
        ),
        max_pico_gap_ms=args.max_pico_gap_ms,
        max_action_age_ms=args.max_action_age_ms,
        max_action_hold_ms=getattr(args, "max_action_hold_ms", 150.0),
        action_offset_ms=getattr(args, "action_offset_ms", 0.0),
        action_match=getattr(args, "action_match", "latest-before"),
        max_clock_fit_residual_ms=args.max_clock_fit_residual_ms,
        max_invalid_ratio=args.max_invalid_ratio,
        max_imputed_ratio=getattr(args, "max_imputed_ratio", 0.10),
        max_sequence_missing_ratio=getattr(
            args, "max_sequence_missing_ratio", 0.05
        ),
        require_source_clocks=args.require_source_clocks,
        require_pico=getattr(args, "require_pico", False),
        require_quality=args.require_quality,
        gap_policy=getattr(args, "gap_policy", "trim-edges"),
        keep_invalid=False,
        overwrite=args.overwrite_aligned,
    )


def build_quality_summary(
    task_dir: Path,
    reports: list[dict[str, Any]],
    failures: list[dict[str, str]],
) -> dict[str, Any]:
    candidate_frames = sum(int(report.get("candidate_frames", 0)) for report in reports)
    valid_frames = sum(int(report.get("valid_frames", 0)) for report in reports)
    episodes = []
    for report in reports:
        camera_p95 = {
            name: values.get("p95")
            for name, values in (report.get("camera_abs_error_ms") or {}).items()
        }
        episodes.append(
            {
                "episode": Path(report["input"]).name,
                "candidate_frames": int(report.get("candidate_frames", 0)),
                "valid_frames": int(report.get("valid_frames", 0)),
                "valid_ratio": float(report.get("valid_ratio", 0.0)),
                "imputed_frames": int(report.get("imputed_frames", 0)),
                "imputed_ratio": float(report.get("imputed_ratio", 0.0)),
                "imputation_counts": report.get("imputation_counts", {}),
                "quality_valid": bool(report.get("quality_valid", False)),
                "quality_issues": report.get("quality_issues", []),
                "quality_warnings": report.get("quality_warnings", []),
                "source_clock_valid": bool(report.get("source_clock_valid", False)),
                "source_clock_issues": report.get("source_clock_issues", []),
                "invalid_counts": report.get("invalid_counts", {}),
                "camera_error_p95_ms": camera_p95,
                "lowstate_gap_p95_ms": (report.get("lowstate_gap_ms") or {}).get("p95"),
                "pico_gap_p95_ms": (report.get("pico_gap_ms") or {}).get("p95"),
                "lowcmd_age_p95_ms": (report.get("lowcmd_age_ms") or {}).get("p95"),
                "timeline": report.get("timeline", {}),
                "action_alignment": report.get("action_alignment", {}),
                "pico_required": bool(report.get("pico_required", False)),
                "raw_queue_dropped": report.get("raw_queue_dropped", {}),
                "raw_write_failed": report.get("raw_write_failed", {}),
            }
        )
    return {
        "task_dir": str(task_dir),
        "episode_count": len(reports),
        "candidate_frames": candidate_frames,
        "valid_frames": valid_frames,
        "valid_ratio": valid_frames / candidate_frames if candidate_frames else 0.0,
        "all_quality_valid": bool(reports) and all(
            report.get("quality_valid", False) for report in reports
        ),
        "all_source_clocks_valid": bool(reports) and all(
            report.get("source_clock_valid", False) for report in reports
        ),
        "failures": failures,
        "episodes": episodes,
    }


def finalize(args: argparse.Namespace) -> Path:
    task_dir = args.task_dir.expanduser().resolve()
    raw_dir = task_dir / "raw"
    raw_episodes = sorted(
        path for path in raw_dir.glob("episode_*") if (path / "manifest.json").exists()
    )
    if not raw_episodes:
        raise FileNotFoundError(f"No raw/episode_XXXX sessions found under {task_dir}")

    aligned_dir = task_dir / "aligned"
    aligned_dir.mkdir(parents=True, exist_ok=True)
    reports: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for raw_episode in raw_episodes:
        output_episode = aligned_dir / raw_episode.name
        try:
            if output_episode.exists() and not args.overwrite_aligned:
                report_path = output_episode / "alignment_report.json"
                if not report_path.exists():
                    raise FileExistsError(
                        f"{output_episode} exists without alignment_report.json; "
                        "use --overwrite-aligned"
                    )
            else:
                align_episode(_alignment_args(args, raw_episode, output_episode))
            reports.append(
                json.loads((output_episode / "alignment_report.json").read_text())
            )
        except Exception as exc:
            failures.append({"episode": raw_episode.name, "error": str(exc)})

    summary = build_quality_summary(task_dir, reports, failures)
    summary_path = task_dir / "offline_quality_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if failures:
        raise RuntimeError(
            f"{len(failures)} episode(s) failed alignment; inspect {summary_path}"
        )

    if args.export_lerobot:
        from teleop.utils.export_lerobot_dataset import write_dataset

        output_dir = (
            args.lerobot_output.expanduser().resolve()
            if args.lerobot_output
            else task_dir / "lerobot_offline"
        )
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", task_dir.name).strip("_")
        write_dataset(
            SimpleNamespace(
                input_dir=aligned_dir,
                output_dir=output_dir,
                repo_id=args.repo_id or f"local/{safe_name or 'tele_robot'}",
                robot_type="TOPSTAR_H1_MUJOCO_IK",
                task_name=None,
                fps=int(round(args.fps)),
                tolerance_s=None,
                no_depth=args.no_depth,
                no_videos=False,
                streaming_encoding=False,
                vcodec=args.video_codec,
                overwrite=args.overwrite_lerobot,
                allow_invalid_alignment=False,
            )
        )
    return summary_path


def main() -> None:
    print(finalize(parse_args()))


if __name__ == "__main__":
    main()
