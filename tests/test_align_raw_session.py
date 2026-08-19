from teleop.utils.align_raw_session import (
    fixed_grid_matches,
    invalid_runs,
    select_action_event,
)


def _event(sequence: int):
    return {"sequence": sequence, "device_timestamp_raw": sequence * 1_000_000}


def test_fixed_grid_keeps_grid_time_instead_of_camera_jitter():
    base = 1_000_000_000
    times = [base, base + 34_000_000, base + 67_000_000, base + 101_000_000]
    matches = fixed_grid_matches([_event(i) for i in range(4)], times, fps=20.0)

    assert [item["target_ns"] for item in matches] == [
        base,
        base + 50_000_000,
        base + 100_000_000,
    ]
    assert matches[1]["event"]["sequence"] == 1
    assert matches[1]["delta_ns"] == -16_000_000


def test_reused_reference_frame_is_explicitly_marked():
    base = 1_000_000_000
    matches = fixed_grid_matches([_event(1)], [base], fps=20.0)
    assert matches[0]["reference_reused"] is False

    # A 5 Hz camera cannot populate a 20 Hz grid without reusing frames.
    times = [base, base + 200_000_000]
    matches = fixed_grid_matches([_event(1), _event(2)], times, fps=20.0)
    assert any(item["reference_reused"] for item in matches[1:])
    assert [item["reference_reuse_run_length"] for item in matches] == [0, 1, 2, 0, 1]


def test_internal_gap_is_distinct_from_trim_safe_edge_gaps():
    runs = invalid_runs([1, 2, 4], candidate_count=6)
    assert runs == [
        {"start_grid_index": 0, "end_grid_index": 0, "length": 1, "touches_edge": True},
        {"start_grid_index": 3, "end_grid_index": 3, "length": 1, "touches_edge": False},
        {"start_grid_index": 5, "end_grid_index": 5, "length": 1, "touches_edge": True},
    ]


def test_action_pairing_semantics_are_explicit():
    events = [{"sequence": 1}, {"sequence": 2}, {"sequence": 3}]
    times = [90, 110, 130]

    event, delta = select_action_event(events, times, 100, "latest-before")
    assert event["sequence"] == 1 and delta == -10
    event, delta = select_action_event(events, times, 100, "first-after")
    assert event["sequence"] == 2 and delta == 10
    event, delta = select_action_event(events, times, 128, "nearest")
    assert event["sequence"] == 3 and delta == 2
