from teleop.utils.check_clock_sync import clock_sync_report


def test_clock_sync_uses_current_system_correction(monkeypatch):
    fields = {
        "Reference ID": "TEST",
        "Stratum": "3",
        "System time": "0.000001000 seconds slow of NTP time",
        "Last offset": "-0.022000000 seconds",
        "Leap status": "Normal",
    }
    monkeypatch.setattr(
        "teleop.utils.check_clock_sync.chronyc_tracking",
        lambda: (fields, None),
    )

    report = clock_sync_report(max_offset_ms=2.0)

    assert report["system_time_offset_ms"] == 0.001
    assert report["last_offset_ms"] == 22.0
    assert report["valid"] is True
