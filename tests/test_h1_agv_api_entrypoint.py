import threading
import time
import types
from pathlib import Path

import numpy as np

from teleop import teleop_h1_arm_head_torso_base as body_profile
from teleop.teleop_h1_arm_head_torso_agv_api import (
    AgvRuntimeConfig,
    LatestAgvVelocityPublisher,
    VendorAgvSession,
    VendorApiModules,
    _install_base_output,
    load_vendor_api,
    parse_runtime_config,
)


class _FakeSocket:
    def __init__(self):
        self.address = None
        self.timeouts = []
        self.sent = []
        self.closed = False

    def settimeout(self, value):
        self.timeouts.append(value)

    def connect(self, address):
        self.address = address

    def sendall(self, payload):
        self.sent.append(payload)

    def shutdown(self, _how):
        pass

    def close(self):
        self.closed = True


class _FakeProtocol:
    def __init__(self):
        self.next_id = 100
        self.requests = []

    def get_num_id(self):
        self.next_id += 1
        return self.next_id

    def packMasg(self, request_id, action, data):
        self.requests.append((request_id, action, data))
        return f"{request_id}:{action}".encode()

    def recvMsg(self, _client):
        request_id, action, _data = self.requests[-1]
        return {"ret_code": 0}, (0x5A, 1, request_id, 0, action, b"\0" * 6)


class _FakeVelocitySession:
    def __init__(self):
        self.connected = False
        self.closed = False
        self.commands = []
        self.command_event = threading.Event()

    def connect(self):
        self.connected = True

    def send_velocity(self, vx, vy, wz, duration_ms):
        self.commands.append((vx, vy, wz, duration_ms, time.monotonic_ns()))
        self.command_event.set()

    def close(self):
        self.closed = True


def _wait_until(predicate, timeout=1.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return False


def test_loads_real_vendor_api_as_isolated_package():
    modules = load_vendor_api(Path("/home/ai/lium/api"))

    assert modules.api.agv_open_loop_motion_process
    assert modules.models.RobotManager
    assert modules.utils.packMasg
    assert modules.api.__name__.startswith("_tele_robot_vendor_agv_api.")


def test_vendor_session_uses_expected_ports_actions_and_units():
    protocol = _FakeProtocol()
    sockets = []

    def socket_factory():
        client = _FakeSocket()
        sockets.append(client)
        return client

    modules = VendorApiModules(
        utils=protocol,
        models=types.SimpleNamespace(create_socket=socket_factory),
        api=types.SimpleNamespace(agv_open_loop_motion_process=lambda: None),
    )
    config = AgvRuntimeConfig(ip="192.168.31.50", control_name="test-pico")
    session = VendorAgvSession(config, modules, socket_factory=socket_factory)

    session.connect()
    session.send_velocity(0.2, -0.1, 0.4, 250)
    session.close()

    assert [client.address for client in sockets] == [
        ("192.168.31.50", 19205),
        ("192.168.31.50", 19207),
    ]
    assert [request[1] for request in protocol.requests] == [4005, 2010, 4006]
    assert protocol.requests[0][2] == {"nick_name": "test-pico"}
    assert protocol.requests[1][2] == {
        "vx": 0.2,
        "vy": -0.1,
        "w": 0.4,
        "duration": 250,
    }
    assert all(client.closed for client in sockets)


def test_latest_publisher_clips_command_and_stops_when_stale():
    config = AgvRuntimeConfig(
        ip="192.168.31.50",
        rate_hz=20.0,
        command_duration_ms=100,
        watchdog_ms=60,
        max_linear_speed_m_s=0.3,
        max_angular_speed_rad_s=0.5,
    )
    session = _FakeVelocitySession()
    publisher = LatestAgvVelocityPublisher(config, lambda: session)

    publisher.start()
    publisher.set_command([1.0, -1.0, 2.0])
    assert _wait_until(
        lambda: any(command[:3] == (0.3, -0.3, 0.5) for command in session.commands)
    )
    nonzero_index = max(
        index
        for index, command in enumerate(session.commands)
        if command[:3] == (0.3, -0.3, 0.5)
    )
    assert _wait_until(
        lambda: any(
            command[:3] == (0.0, 0.0, 0.0)
            for command in session.commands[nonzero_index + 1 :]
        )
    )

    publisher.close()

    assert session.connected
    assert session.closed
    assert session.commands[-1][:4] == (0.0, 0.0, 0.0, 100)


def test_custom_arguments_are_removed_before_old_program_parser():
    config, forwarded = parse_runtime_config(
        [
            "teleop_h1_arm_head_torso_agv_api.py",
            "--agv-ip",
            "192.168.31.50",
            "--agv-rate",
            "5",
            "--agv-command-duration-ms",
            "400",
            "--frequency",
            "20",
            "--record",
        ]
    )

    assert config.ip == "192.168.31.50"
    assert config.rate_hz == 5.0
    assert config.command_duration_ms == 400
    assert forwarded == [
        "teleop_h1_arm_head_torso_agv_api.py",
        "--frequency",
        "20",
        "--record",
    ]


def test_base_output_patch_is_process_local_and_reversible():
    commands = []
    publisher = types.SimpleNamespace(
        set_command=lambda command: commands.append(np.asarray(command).copy())
    )
    publisher.config = types.SimpleNamespace(
        max_linear_speed_m_s=0.4,
        max_angular_speed_rad_s=0.6,
    )
    original = _install_base_output(publisher, source_scale=0.5)
    try:
        body_profile.H1BodyJoystickController._publish_base(
            object(), np.array([0.1, 0.2, 0.3])
        )
    finally:
        body_profile.H1BodyJoystickController._publish_base = staticmethod(original)

    assert len(commands) == 1
    np.testing.assert_allclose(commands[0], [0.08, 0.16, 0.36])
    assert body_profile.H1BodyJoystickController._publish_base is original
