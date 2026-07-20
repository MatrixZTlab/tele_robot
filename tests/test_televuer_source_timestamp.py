from datetime import datetime
from multiprocessing import Queue, Value
from types import SimpleNamespace

from televuer.televuer import TeleVuer
from vuer.events import ClientEvent


def test_client_event_timestamp_falls_back_to_vuer_top_level_ts():
    timestamp_ms = 1_784_100_123_456.0
    event = ClientEvent(
        etype="CONTROLLER_MOVE",
        value={"left": [], "right": []},
        ts=timestamp_ms,
    )

    raw, key = TeleVuer._source_timestamp_from_client_event(event)

    assert raw == timestamp_ms
    assert key == "event.ts"


def test_pose_sample_timestamp_takes_priority_over_send_timestamp():
    event = SimpleNamespace(
        value={"predictedDisplayTime": 123_456.75},
        ts=datetime.fromtimestamp(1_784_100_123.456),
    )

    raw, key = TeleVuer._source_timestamp_from_client_event(event)

    assert raw == 123_456.75
    assert key == "predictedDisplayTime"


def test_numeric_vuer_timestamp_is_normalized_to_milliseconds():
    event = SimpleNamespace(value={}, ts=1_784_100_123.456)

    raw, key = TeleVuer._source_timestamp_from_client_event(event)

    assert raw == 1_784_100_123_456.0
    assert key == "event.ts"


def test_source_timestamp_is_written_to_raw_xr_queue():
    tvuer = TeleVuer.__new__(TeleVuer)
    tvuer.xr_event_drop_count_shared = Value("q", 0, lock=True)
    tvuer.xr_event_queue = Queue(maxsize=2)

    try:
        tvuer._enqueue_xr_event(
            "controller",
            {"left": [1.0] * 16, "right": [2.0] * 16},
            monotonic_ns=123,
            wall_ns=456,
            sequence=7,
            source_timestamp_raw=1_784_100_123_456.0,
            source_timestamp_key="event.ts",
        )
        payload = tvuer.xr_event_queue.get(timeout=1.0)
    finally:
        tvuer.xr_event_queue.close()
        tvuer.xr_event_queue.join_thread()

    assert payload["source_timestamp_raw"] == 1_784_100_123_456.0
    assert payload["source_timestamp_key"] == "event.ts"
    assert payload["host_receive_monotonic_ns"] == 123
    assert payload["host_receive_wall_ns"] == 456
