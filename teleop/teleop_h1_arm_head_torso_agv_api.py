"""TOPSTAR H1 upper-body teleoperation with the vendor AGV TCP API.

This is a separate entry point.  It reuses the existing arm/head/recording
program and redirects only the optional body-control profile's base output to
the API under ``/home/ai/lium/api``.

Controller mapping:

* Left stick Y/X: AGV forward/lateral velocity.
* Left stick click + X: AGV yaw velocity instead of lateral velocity.
* Right stick Y: torso pitch velocity.
* Right stick X: torso lift velocity (the H1 model has no torso-yaw joint).

The AGV sender is intentionally isolated from the IK loop.  It sends only the
latest command, uses the API command duration as a device-side watchdog, and
sends a final zero command before releasing control.
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import os
import runpy
import socket
import sys
import threading
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np


CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teleop import teleop_h1_arm_head_torso_base as body_profile


LOGGER = logging.getLogger("h1_agv_api_teleop")
VENDOR_PACKAGE = "_tele_robot_vendor_agv_api"


@dataclass(frozen=True)
class AgvRuntimeConfig:
    ip: str
    api_dir: Path = Path("/home/ai/lium/api")
    rate_hz: float = 10.0
    command_duration_ms: int = 250
    watchdog_ms: int = 150
    connect_timeout_s: float = 3.0
    response_timeout_s: float = 1.0
    max_linear_speed_m_s: float = 0.30
    max_angular_speed_rad_s: float = 0.50
    acquire_control_lock: bool = True
    control_name: str = "tele-robot-pico"

    def validate(self) -> None:
        if not self.ip.strip():
            raise ValueError("AGV IP must not be empty")
        if not self.api_dir.is_dir():
            raise ValueError(f"AGV API directory does not exist: {self.api_dir}")
        for filename in ("utils.py", "models.py", "AGV_api.py"):
            if not (self.api_dir / filename).is_file():
                raise ValueError(f"AGV API file is missing: {self.api_dir / filename}")
        if not 1.0 <= self.rate_hz <= 20.0:
            raise ValueError("AGV send rate must be in [1, 20] Hz")
        if not 50 <= self.command_duration_ms <= 1000:
            raise ValueError("AGV command duration must be in [50, 1000] ms")
        if self.command_duration_ms < 2.0 * 1000.0 / self.rate_hz:
            raise ValueError("AGV command duration must cover at least two send periods")
        if not 50 <= self.watchdog_ms <= 500:
            raise ValueError("AGV watchdog must be in [50, 500] ms")
        if self.connect_timeout_s <= 0 or self.response_timeout_s <= 0:
            raise ValueError("AGV socket timeouts must be positive")
        if self.max_linear_speed_m_s <= 0 or self.max_angular_speed_rad_s <= 0:
            raise ValueError("AGV speed limits must be positive")
        if not self.control_name.strip():
            raise ValueError("AGV control name must not be empty")


@dataclass(frozen=True)
class VendorApiModules:
    utils: types.ModuleType
    models: types.ModuleType
    api: types.ModuleType


def _load_module(fullname: str, path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(fullname, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load {fullname} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[fullname] = module
    spec.loader.exec_module(module)
    return module


def load_vendor_api(api_dir: Path) -> VendorApiModules:
    """Load the loose vendor files as an isolated package."""
    api_dir = api_dir.resolve()
    existing = sys.modules.get(VENDOR_PACKAGE)
    if existing is not None and Path(existing.__path__[0]).resolve() != api_dir:
        for name in tuple(sys.modules):
            if name == VENDOR_PACKAGE or name.startswith(VENDOR_PACKAGE + "."):
                del sys.modules[name]
        existing = None

    if existing is None:
        package = types.ModuleType(VENDOR_PACKAGE)
        package.__path__ = [str(api_dir)]
        package.__package__ = VENDOR_PACKAGE
        sys.modules[VENDOR_PACKAGE] = package

    utils = sys.modules.get(f"{VENDOR_PACKAGE}.utils")
    if utils is None:
        utils = _load_module(f"{VENDOR_PACKAGE}.utils", api_dir / "utils.py")
    models = sys.modules.get(f"{VENDOR_PACKAGE}.models")
    if models is None:
        models = _load_module(f"{VENDOR_PACKAGE}.models", api_dir / "models.py")
    api = sys.modules.get(f"{VENDOR_PACKAGE}.AGV_api")
    if api is None:
        api = _load_module(f"{VENDOR_PACKAGE}.AGV_api", api_dir / "AGV_api.py")

    if not hasattr(api, "agv_open_loop_motion_process"):
        raise ImportError("Vendor API has no agv_open_loop_motion_process()")
    return VendorApiModules(utils=utils, models=models, api=api)


class VendorAgvSession:
    """Small synchronous client using the framing supplied by the vendor API."""

    CONTROL_PORT = 19205
    CONFIG_PORT = 19207
    OPEN_LOOP_ACTION = 2010
    LOCK_ACTION = 4005
    UNLOCK_ACTION = 4006

    def __init__(
        self,
        config: AgvRuntimeConfig,
        modules: VendorApiModules,
        socket_factory: Callable[[], socket.socket] | None = None,
    ):
        self.config = config
        self.modules = modules
        self._socket_factory = socket_factory or modules.models.create_socket
        self._control_socket: socket.socket | None = None
        self._config_socket: socket.socket | None = None
        self._control_locked = False

    def _connect_socket(self, port: int) -> socket.socket:
        client = self._socket_factory()
        client.settimeout(self.config.connect_timeout_s)
        try:
            client.connect((self.config.ip, port))
            client.settimeout(self.config.response_timeout_s)
        except Exception:
            client.close()
            raise
        return client

    def connect(self) -> None:
        self._control_socket = self._connect_socket(self.CONTROL_PORT)
        try:
            if self.config.acquire_control_lock:
                self._config_socket = self._connect_socket(self.CONFIG_PORT)
                self._request(
                    self._config_socket,
                    self.LOCK_ACTION,
                    {"nick_name": self.config.control_name},
                )
                self._control_locked = True
        except Exception:
            self.close()
            raise

    def _request(self, client: socket.socket, action: int, data: dict[str, Any]) -> dict:
        request_id = int(self.modules.utils.get_num_id())
        payload = self.modules.utils.packMasg(request_id, int(action), data)
        client.sendall(payload)
        result, header = self.modules.utils.recvMsg(client)
        if len(header) < 5 or int(header[2]) != request_id:
            raise RuntimeError(
                f"AGV response request id mismatch: expected {request_id}, got {header}"
            )
        if not isinstance(result, dict):
            raise RuntimeError(f"AGV action {action} returned invalid data: {result!r}")
        ret_code = int(result.get("ret_code", 0))
        if ret_code != 0:
            raise RuntimeError(
                f"AGV action {action} failed: ret_code={ret_code}, "
                f"err_msg={result.get('err_msg', '')}"
            )
        return result

    def send_velocity(self, vx: float, vy: float, wz: float, duration_ms: int) -> None:
        if self._control_socket is None:
            raise RuntimeError("AGV control socket is not connected")
        self._request(
            self._control_socket,
            self.OPEN_LOOP_ACTION,
            {
                "vx": float(vx),
                "vy": float(vy),
                "w": float(wz),
                "duration": int(duration_ms),
            },
        )

    def close(self) -> None:
        if self._control_locked and self._config_socket is not None:
            try:
                self._request(self._config_socket, self.UNLOCK_ACTION, {})
            except Exception as exc:
                LOGGER.warning("Failed to release AGV control lock: %s", exc)
        self._control_locked = False
        for client in (self._control_socket, self._config_socket):
            if client is not None:
                try:
                    client.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                try:
                    client.close()
                except OSError:
                    pass
        self._control_socket = None
        self._config_socket = None


class LatestAgvVelocityPublisher:
    """Fixed-rate AGV sender with latest-value semantics and stale-command stop."""

    def __init__(
        self,
        config: AgvRuntimeConfig,
        session_factory: Callable[[], Any],
    ):
        self.config = config
        self._session_factory = session_factory
        self._condition = threading.Condition()
        self._latest = np.zeros(3, dtype=float)
        self._latest_ns = 0
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._session: Any = None
        self._failure: BaseException | None = None

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("AGV publisher is already started")
        self._session = self._session_factory()
        self._session.connect()
        try:
            self._session.send_velocity(
                0.0, 0.0, 0.0, self.config.command_duration_ms
            )
        except Exception:
            self._session.close()
            self._session = None
            raise
        self._thread = threading.Thread(
            target=self._run,
            name="agv-latest-command",
            daemon=True,
        )
        self._thread.start()

    def set_command(self, command: np.ndarray | list[float]) -> None:
        self.raise_if_failed()
        value = np.asarray(command, dtype=float).reshape(-1)
        if value.size != 3 or not np.all(np.isfinite(value)):
            raise ValueError("AGV command must contain three finite values")
        value = value.copy()
        value[:2] = np.clip(
            value[:2],
            -self.config.max_linear_speed_m_s,
            self.config.max_linear_speed_m_s,
        )
        value[2] = np.clip(
            value[2],
            -self.config.max_angular_speed_rad_s,
            self.config.max_angular_speed_rad_s,
        )
        with self._condition:
            self._latest = value
            self._latest_ns = time.monotonic_ns()
            self._condition.notify_all()

    def raise_if_failed(self) -> None:
        if self._failure is not None:
            raise RuntimeError("AGV command thread failed") from self._failure

    def _current_command(self, now_ns: int) -> np.ndarray:
        with self._condition:
            age_ms = (now_ns - self._latest_ns) / 1e6 if self._latest_ns else float("inf")
            if age_ms > self.config.watchdog_ms:
                return np.zeros(3, dtype=float)
            return self._latest.copy()

    def _run(self) -> None:
        period_s = 1.0 / self.config.rate_hz
        deadline = time.monotonic()
        try:
            while not self._stop_event.is_set():
                command = self._current_command(time.monotonic_ns())
                self._session.send_velocity(
                    float(command[0]),
                    float(command[1]),
                    float(command[2]),
                    self.config.command_duration_ms,
                )
                deadline += period_s
                wait_s = deadline - time.monotonic()
                if wait_s <= 0:
                    deadline = time.monotonic()
                    continue
                self._stop_event.wait(wait_s)
        except BaseException as exc:
            self._failure = exc
            LOGGER.exception("AGV command thread stopped after a communication error")

    def close(self) -> None:
        with self._condition:
            self._latest[:] = 0.0
            self._latest_ns = time.monotonic_ns()
        self._stop_event.set()
        thread_alive = False
        if self._thread is not None:
            self._thread.join(timeout=self.config.response_timeout_s + 1.0)
            thread_alive = self._thread.is_alive()
        if self._session is not None:
            if thread_alive:
                LOGGER.error(
                    "AGV command thread did not stop; closing sockets and relying on "
                    "the %d ms device command timeout",
                    self.config.command_duration_ms,
                )
                self._session.close()
                self._session = None
                self._thread = None
                return
            try:
                self._session.send_velocity(
                    0.0, 0.0, 0.0, self.config.command_duration_ms
                )
                LOGGER.info("AGV final zero-velocity command acknowledged")
            except Exception as exc:
                LOGGER.error(
                    "AGV final zero command failed; the %d ms device command must expire: %s",
                    self.config.command_duration_ms,
                    exc,
                )
            finally:
                self._session.close()
        self._session = None
        self._thread = None


def _custom_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--agv-ip")
    parser.add_argument("--agv-api-dir", default="/home/ai/lium/api")
    parser.add_argument("--agv-rate", type=float, default=10.0)
    parser.add_argument("--agv-command-duration-ms", type=int, default=250)
    parser.add_argument("--agv-watchdog-ms", type=int, default=150)
    parser.add_argument("--agv-connect-timeout-s", type=float, default=3.0)
    parser.add_argument("--agv-response-timeout-s", type=float, default=1.0)
    parser.add_argument("--agv-max-linear-speed", type=float, default=0.30)
    parser.add_argument("--agv-max-angular-speed", type=float, default=0.50)
    parser.add_argument("--agv-no-control-lock", action="store_true")
    parser.add_argument("--agv-control-name", default="tele-robot-pico")
    return parser


def parse_runtime_config(argv: list[str]) -> tuple[AgvRuntimeConfig, list[str]]:
    custom, remaining = _custom_parser().parse_known_args(argv[1:])
    if custom.agv_ip is None:
        raise SystemExit(
            "--agv-ip is required; use the AGV controller IP, not the ROS2 computer IP"
        )
    config = AgvRuntimeConfig(
        ip=custom.agv_ip,
        api_dir=Path(custom.agv_api_dir),
        rate_hz=custom.agv_rate,
        command_duration_ms=custom.agv_command_duration_ms,
        watchdog_ms=custom.agv_watchdog_ms,
        connect_timeout_s=custom.agv_connect_timeout_s,
        response_timeout_s=custom.agv_response_timeout_s,
        max_linear_speed_m_s=custom.agv_max_linear_speed,
        max_angular_speed_rad_s=custom.agv_max_angular_speed,
        acquire_control_lock=not custom.agv_no_control_lock,
        control_name=custom.agv_control_name,
    )
    config.validate()
    return config, [argv[0], *remaining]


def _body_source_scale(argv: list[str]) -> float:
    value = body_profile._option_value(argv, "--body-max-speed")
    requested = 1.0 if value is None else abs(float(value))
    return float(np.clip(requested, 0.0, 1.0))


def _install_base_output(
    publisher: LatestAgvVelocityPublisher,
    source_scale: float = 1.0,
):
    original = body_profile.H1BodyJoystickController._publish_base
    denominator = max(float(source_scale), np.finfo(float).eps)

    def publish_to_agv(_arm_ctrl: Any, command: np.ndarray) -> None:
        # The shared body profile emits all axes using body_max_speed.  Convert
        # that value back to normalized stick travel, then apply physical AGV
        # limits per axis.  Mutating this local command copy also makes the raw
        # base_command event contain the physical values sent to the AGV.
        command[:2] = (
            np.clip(command[:2] / denominator, -1.0, 1.0)
            * publisher.config.max_linear_speed_m_s
        )
        command[2] = (
            np.clip(command[2] / denominator, -1.0, 1.0)
            * publisher.config.max_angular_speed_rad_s
        )
        publisher.set_command(command)

    body_profile.H1BodyJoystickController._publish_base = staticmethod(publish_to_agv)
    return original


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config, forwarded_argv = parse_runtime_config(sys.argv)
    modules = load_vendor_api(config.api_dir)
    publisher = LatestAgvVelocityPublisher(
        config,
        session_factory=lambda: VendorAgvSession(config, modules),
    )

    body_profile._prepare_argv(forwarded_argv)
    os.environ[body_profile.PROFILE_ENV] = body_profile.PROFILE_NAME
    sys.argv = forwarded_argv
    original_publish = _install_base_output(
        publisher,
        source_scale=_body_source_scale(forwarded_argv),
    )

    LOGGER.info(
        "AGV API target=%s ports=%d/%d rate=%.1fHz duration=%dms watchdog=%dms",
        config.ip,
        VendorAgvSession.CONTROL_PORT,
        VendorAgvSession.CONFIG_PORT,
        config.rate_hz,
        config.command_duration_ms,
        config.watchdog_ms,
    )
    LOGGER.info(
        "Pico controls: left stick=AGV XY, left-stick click+X=AGV yaw; "
        "right stick Y=torso pitch, X=torso lift"
    )

    try:
        publisher.start()
        runpy.run_path(
            str(CURRENT_DIR / "teleop_hand_and_arm.py"),
            run_name="__main__",
        )
        publisher.raise_if_failed()
    finally:
        body_profile.H1BodyJoystickController._publish_base = staticmethod(
            original_publish
        )
        publisher.close()


if __name__ == "__main__":
    main()
