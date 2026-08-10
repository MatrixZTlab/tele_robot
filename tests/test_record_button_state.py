import time
from types import SimpleNamespace

import teleop.teleop_hand_and_arm as teleop_main
from teleop.utils.episode_writer import EpisodeWriter
from teleop.utils.raw_session_writer import RawSessionWriter


def _wait_until_ready(writer, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if writer.is_ready():
            return
        time.sleep(0.01)
    raise AssertionError("EpisodeWriter did not become ready")


class FakeRawWriter:
    def __init__(self, discard_error=None):
        self.started = []
        self.discarded = []
        self.discard_error = discard_error

    def start_episode(self, episode_index, metadata=None):
        self.started.append((episode_index, metadata))

    def stop_episode(self):
        return None

    def discard_episode(self, episode_index):
        self.discarded.append(episode_index)
        if self.discard_error is not None:
            raise self.discard_error
        return f"raw/episode_{episode_index:04d}"


class FakeHomeRobot:
    def __init__(self):
        self.reset_count = 0
        self.home_count = 0
        self.reference_values = []

    def reset_dual_object_control(self):
        self.reset_count += 1

    def go_home(self):
        self.home_count += 1

    def _reset_arm_references(self, value):
        self.reference_values.append(list(value))


class FakeHomeArmController:
    def __init__(self):
        self.wait_timeouts = []

    def wait_for_hold_expire(self, timeout):
        self.wait_timeouts.append(timeout)

    def get_current_dual_arm_q(self):
        return [0.1] * 14


def _configure_state(monkeypatch, writer, raw_writer):
    args = SimpleNamespace(
        record=True,
        task_name="test",
        task_goal="test",
        frequency=20.0,
        robot="TOPSTAR_H1",
        control_mode="arms_only",
        input_mode="controller",
        img_server_ip="192.168.31.3",
    )
    monkeypatch.setattr(teleop_main, "args", args, raising=False)
    monkeypatch.setattr(teleop_main, "camera_config", {}, raising=False)
    monkeypatch.setattr(teleop_main, "episode_writer", writer, raising=False)
    monkeypatch.setattr(teleop_main, "raw_writer", raw_writer, raising=False)
    monkeypatch.setattr(teleop_main, "READY", True)
    monkeypatch.setattr(teleop_main, "RECORD_RUNNING", False)
    monkeypatch.setattr(teleop_main, "RECORD_TOGGLE", False)
    monkeypatch.setattr(teleop_main, "RECORD_DISCARD", False)
    monkeypatch.setattr(teleop_main, "LAST_RECORD_EPISODE_ID", None)


def test_right_a_is_one_toggle_per_press(monkeypatch):
    wrapper = SimpleNamespace()
    monkeypatch.setattr(teleop_main, "tv_wrapper", wrapper, raising=False)
    released = SimpleNamespace(right_ctrl_aButton=False)
    pressed = SimpleNamespace(right_ctrl_aButton=True)

    assert teleop_main._consume_right_a_press(released) is False
    assert teleop_main._consume_right_a_press(pressed) is True
    assert teleop_main._consume_right_a_press(pressed) is False
    assert teleop_main._consume_right_a_press(released) is False
    assert teleop_main._consume_right_a_press(pressed) is True


def test_left_squeeze_returns_home_once_after_release(monkeypatch):
    robot = FakeHomeRobot()
    arm = FakeHomeArmController()
    monkeypatch.setattr(teleop_main, "robot", robot, raising=False)
    monkeypatch.setattr(teleop_main, "arm_ctrl", arm, raising=False)
    monkeypatch.setattr(teleop_main, "TELEOP_ACTIVE", False)
    monkeypatch.setattr(teleop_main, "RECORD_RUNNING", False)
    monkeypatch.setattr(teleop_main, "_home_left_squeeze_armed", False)

    released = SimpleNamespace(left_ctrl_squeeze=False, left_ctrl_squeezeValue=0.0)
    pressed = SimpleNamespace(left_ctrl_squeeze=True, left_ctrl_squeezeValue=1.0)

    assert teleop_main._process_home_squeeze(pressed) is False
    assert teleop_main._process_home_squeeze(released) is False
    assert teleop_main._process_home_squeeze(pressed) is True
    assert teleop_main._process_home_squeeze(pressed) is False

    assert robot.reset_count == 1
    assert robot.home_count == 1
    assert robot.reference_values == [[0.1] * 14]
    assert arm.wait_timeouts == [5.0]


def test_left_squeeze_home_is_rejected_while_recording(monkeypatch):
    robot = FakeHomeRobot()
    arm = FakeHomeArmController()
    monkeypatch.setattr(teleop_main, "robot", robot, raising=False)
    monkeypatch.setattr(teleop_main, "arm_ctrl", arm, raising=False)
    monkeypatch.setattr(teleop_main, "TELEOP_ACTIVE", False)
    monkeypatch.setattr(teleop_main, "RECORD_RUNNING", True)
    monkeypatch.setattr(teleop_main, "_home_left_squeeze_armed", True)

    pressed = SimpleNamespace(left_ctrl_squeeze=True, left_ctrl_squeezeValue=1.0)
    assert teleop_main._process_home_squeeze(pressed) is False
    assert robot.home_count == 0


def test_discard_cannot_remove_episode_from_previous_process(monkeypatch, tmp_path):
    old_episode = tmp_path / "task" / "episode_0004"
    old_episode.mkdir(parents=True)
    (old_episode / "data.json").write_text("old", encoding="utf-8")
    writer = EpisodeWriter(tmp_path / "task", rerun_log=False)
    raw_writer = FakeRawWriter()
    _configure_state(monkeypatch, writer, raw_writer)

    teleop_main.RECORD_DISCARD = True
    teleop_main._process_record_requests()

    assert old_episode.exists()
    assert raw_writer.discarded == []
    writer.close()


def test_start_then_discard_removes_current_process_episode(monkeypatch, tmp_path):
    writer = EpisodeWriter(tmp_path / "task", rerun_log=False)
    raw_writer = FakeRawWriter()
    _configure_state(monkeypatch, writer, raw_writer)

    teleop_main.RECORD_TOGGLE = True
    teleop_main._process_record_requests()
    episode_index = teleop_main.LAST_RECORD_EPISODE_ID
    episode_dir = tmp_path / "task" / f"episode_{episode_index:04d}"
    assert episode_dir.exists()
    assert teleop_main.RECORD_RUNNING is True

    teleop_main.RECORD_DISCARD = True
    teleop_main._process_record_requests()
    _wait_until_ready(writer)

    assert not episode_dir.exists()
    assert raw_writer.discarded == [episode_index]
    assert teleop_main.RECORD_RUNNING is False
    assert teleop_main.LAST_RECORD_EPISODE_ID is None
    writer.close()


def test_raw_discard_failure_preserves_online_episode(monkeypatch, tmp_path):
    writer = EpisodeWriter(tmp_path / "task", rerun_log=False)
    raw_writer = FakeRawWriter(discard_error=TimeoutError("flush timeout"))
    _configure_state(monkeypatch, writer, raw_writer)

    teleop_main.RECORD_TOGGLE = True
    teleop_main._process_record_requests()
    episode_index = teleop_main.LAST_RECORD_EPISODE_ID
    episode_dir = tmp_path / "task" / f"episode_{episode_index:04d}"

    teleop_main.RECORD_DISCARD = True
    teleop_main._process_record_requests()
    _wait_until_ready(writer)

    assert episode_dir.exists()
    assert teleop_main.RECORD_RUNNING is False
    assert teleop_main.LAST_RECORD_EPISODE_ID is None
    writer.close()


def test_y_save_then_b_discards_online_and_raw(monkeypatch, tmp_path):
    task_dir = tmp_path / "task"
    writer = EpisodeWriter(task_dir, rerun_log=False)
    raw_writer = RawSessionWriter(task_dir, queue_size=16)
    _configure_state(monkeypatch, writer, raw_writer)

    teleop_main.RECORD_TOGGLE = True
    teleop_main._process_record_requests()
    episode_index = teleop_main.LAST_RECORD_EPISODE_ID
    online_dir = task_dir / f"episode_{episode_index:04d}"
    raw_dir = task_dir / "raw" / f"episode_{episode_index:04d}"
    writer.add_item(colors={}, states={"q": [0.1]}, actions={"q": [0.2]})
    assert raw_writer.append_event("lowstate", {"sequence": 1})

    teleop_main.RECORD_TOGGLE = True
    teleop_main._process_record_requests()
    _wait_until_ready(writer)
    assert online_dir.exists()
    assert raw_dir.exists()

    teleop_main.RECORD_DISCARD = True
    teleop_main._process_record_requests()
    _wait_until_ready(writer)

    assert not online_dir.exists()
    assert not raw_dir.exists()
    writer.close()
    raw_writer.close()
