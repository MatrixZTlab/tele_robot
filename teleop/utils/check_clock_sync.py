#!/usr/bin/env python3
"""Check whether this host is sufficiently synchronized for cross-host recording."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time


def chronyc_tracking() -> tuple[dict[str, str], str | None]:
    try:
        result = subprocess.run(
            ["chronyc", "tracking"], capture_output=True, text=True,
            timeout=5.0, check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return {}, str(exc)
    if result.returncode != 0:
        return {}, result.stderr.strip() or result.stdout.strip()
    fields = {}
    for line in result.stdout.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            fields[key.strip()] = value.strip()
    return fields, None


def seconds_field(fields: dict[str, str], key: str) -> float | None:
    value = fields.get(key)
    if not value:
        return None
    match = re.search(r"([-+]?\d+(?:\.\d+)?)\s+seconds", value)
    return float(match.group(1)) if match else None


def clock_sync_report(max_offset_ms: float = 2.0) -> dict[str, object]:
    fields, error = chronyc_tracking()
    system_time_s = seconds_field(fields, "System time")
    last_offset_s = seconds_field(fields, "Last offset")
    report = {
        "wall_ns": time.time_ns(),
        "monotonic_ns": time.monotonic_ns(),
        "chrony_error": error,
        "reference_id": fields.get("Reference ID"),
        "stratum": fields.get("Stratum"),
        "leap_status": fields.get("Leap status"),
        "system_time_offset_ms": (
            abs(system_time_s) * 1000 if system_time_s is not None else None
        ),
        "last_offset_ms": (
            abs(last_offset_s) * 1000 if last_offset_s is not None else None
        ),
        "max_offset_ms": max_offset_ms,
    }
    report["valid"] = (
        error is None
        and report["system_time_offset_ms"] is not None
        and report["system_time_offset_ms"] <= max_offset_ms
        and fields.get("Leap status", "").lower() == "normal"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-offset-ms", type=float, default=2.0)
    args = parser.parse_args()
    report = clock_sync_report(args.max_offset_ms)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["valid"] else 2)


if __name__ == "__main__":
    main()
