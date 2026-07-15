import sys
from pathlib import Path

import numpy as np
import cv2


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "teleop" / "teleimager" / "src"))

from teleimager.image_client import (  # noqa: E402
    SimpleFPSMonitor,
    TripleRingBuffer,
    ZMQ_SubscriberThread,
)


def test_decode_image_and_packet_listener_are_independent():
    subscriber = object.__new__(ZMQ_SubscriberThread)
    subscriber._listener_lock = __import__("threading").Lock()
    subscriber._packet_listeners = []
    received = []
    subscriber.add_packet_listener(received.append)

    image = np.full((3, 4, 3), 73, dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    decoded = subscriber._decode_image(encoded.tobytes())
    subscriber._notify_packet_listeners({"sequence": 9})

    assert decoded.shape == image.shape
    assert received == [{"sequence": 9}]


def test_recv_keeps_decoded_image_and_timestamp_from_same_packet():
    subscriber = object.__new__(ZMQ_SubscriberThread)
    subscriber._request_bgr = True
    subscriber._fps_monitor = SimpleFPSMonitor(window_size=10)
    subscriber._frame_3ring_buffer = TripleRingBuffer()
    subscriber._bgr_3ring_buffer = TripleRingBuffer()

    old_bgr = np.full((2, 3, 3), 17, dtype=np.uint8)
    subscriber._bgr_3ring_buffer.write({
        "jpg": b"frame-1",
        "bgr": old_bgr,
        "timestamp_ns": 101,
        "received_monotonic_ns": 201,
        "received_wall_ns": 301,
    })
    subscriber._frame_3ring_buffer.write({
        "jpg": b"frame-2",
        "timestamp_ns": 102,
        "received_monotonic_ns": 202,
        "received_wall_ns": 302,
    })

    image = subscriber.recv()

    assert image.jpg == b"frame-1"
    assert np.array_equal(image.bgr, old_bgr)
    assert image.timestamp_ns == 101
    assert image.received_monotonic_ns == 201
    assert image.received_wall_ns == 301


def test_recv_exposes_receipt_timestamps_without_decoding():
    subscriber = object.__new__(ZMQ_SubscriberThread)
    subscriber._request_bgr = False
    subscriber._fps_monitor = SimpleFPSMonitor(window_size=10)
    subscriber._frame_3ring_buffer = TripleRingBuffer()
    subscriber._bgr_3ring_buffer = None
    subscriber._frame_3ring_buffer.write({
        "jpg": b"frame",
        "timestamp_ns": 11,
        "received_monotonic_ns": 22,
        "received_wall_ns": 33,
    })

    image = subscriber.recv()

    assert image.timestamp_ns == 11
    assert image.received_monotonic_ns == 22
    assert image.received_wall_ns == 33
