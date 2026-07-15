import json

import numpy as np

from teleop.utils.analyze_teleop_latency import (
    _best_feedback_lag_ms,
    _load_alignment,
)


def test_load_alignment_prefers_monotonic_timestamps(tmp_path):
    path = tmp_path / "alignment.jsonl"
    row = {
        "alignment": {
            "actual_timestamps_ns": {
                "control_start_monotonic": 100,
                "robot_step_done_monotonic": 140,
                "control_cycle_wall": 1_000,
                "robot_step_done_wall": 1_050,
            }
        }
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    control, done = _load_alignment(path)

    np.testing.assert_array_equal(control, [100])
    np.testing.assert_array_equal(done, [140])


def test_feedback_lag_caps_requested_lag_to_available_frames():
    actions = np.arange(20, dtype=float).reshape(10, 2)
    states = actions.copy()
    control_ns = np.arange(10, dtype=np.int64) * 50_000_000

    lag, lag_ms, mae = _best_feedback_lag_ms(
        states, actions, control_ns, max_lag=100
    )

    assert lag == 0
    assert lag_ms == 0.0
    assert mae == 0.0
