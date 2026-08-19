import json
from types import SimpleNamespace

import teleop.teleop_hand_and_arm as teleop_main
from teleop.utils.raw_session_writer import RawSessionWriter


class FakeRawWriter:
    def __init__(self):
        self.started = []
        self.stopped = 0
        self.discarded = []
        self.active = False

    def start_episode(self, episode_index, metadata=None):
        self.started.append((episode_index, metadata))
        self.active = True

    def stop_episode(self):
        self.stopped += 1
        self.active = False
        return "raw/episode"

    def discard_episode(self, episode_index):
        self.discarded.append(episode_index)
        self.active = False
        return f"raw/episode_{episode_index:04d}"


class FakeHomeRobot:
    def __init__(self):
        self.reset_count = 0
        self.relative_reset_count = 0
        self.home_count = 0
        self.reference_values = []

    def reset_dual_object_control(self):
        self.reset_count += 1

    def reset_relative_pose_state(self, side):
        assert side == "both"
        self.relative_reset_count += 1

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


def _configure_state(monkeypatch, raw_writer):
    args = SimpleNamespace(
        record=True,
        task_name="test",
        task_goal="test",
        task_desc="test description",
        frequency=20.0,
        robot="TOPSTAR_H1",
        control_mode="arms_only",
        input_mode="controller",
        img_server_ip="192.168.31.3",
        controller_mapping="same-side",
        orientation_control="locked",
    )
    monkeypatch.setattr(teleop_main, "args", args, raising=False)
    monkeypatch.setattr(teleop_main, "camera_config", {}, raising=False)
    monkeypatch.setattr(teleop_main, "raw_writer", raw_writer, raising=False)
    monkeypatch.setattr(
        teleop_main, "_missing_raw_camera_frames", lambda: [], raising=False
    )
    monkeypatch.setattr(teleop_main, "READY", True)
    monkeypatch.setattr(teleop_main, "RECORD_RUNNING", False)
    monkeypatch.setattr(teleop_main, "RECORD_TOGGLE", False)
    monkeypatch.setattr(teleop_main, "RECORD_DISCARD", False)
    monkeypatch.setattr(teleop_main, "LAST_RECORD_EPISODE_ID", None)
    monkeypatch.setattr(teleop_main, "CURRENT_RECORD_EPISODE_ID", None)
    monkeypatch.setattr(teleop_main, "NEXT_RECORD_EPISODE_ID", 0)
    monkeypatch.setattr(teleop_main, "TELEOP_MODE", "relative_pose")


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
    wrapper = SimpleNamespace(
        reset_left_wrist_ref=lambda: None,
        reset_right_wrist_ref=lambda: None,
    )
    monkeypatch.setattr(teleop_main, "robot", robot, raising=False)
    monkeypatch.setattr(teleop_main, "arm_ctrl", arm, raising=False)
    monkeypatch.setattr(teleop_main, "tv_wrapper", wrapper, raising=False)
    monkeypatch.setattr(teleop_main, "TELEOP_ACTIVE", False)
    monkeypatch.setattr(teleop_main, "LEFT_ARM_ACTIVE", False)
    monkeypatch.setattr(teleop_main, "RIGHT_ARM_ACTIVE", False)
    monkeypatch.setattr(teleop_main, "RECORD_RUNNING", False)
    monkeypatch.setattr(teleop_main, "_home_left_squeeze_armed", False)

    released = SimpleNamespace(left_ctrl_squeeze=False, left_ctrl_squeezeValue=0.0)
    pressed = SimpleNamespace(left_ctrl_squeeze=True, left_ctrl_squeezeValue=1.0)

    assert teleop_main._process_home_squeeze(pressed) is False
    assert teleop_main._process_home_squeeze(released) is False
    assert teleop_main._process_home_squeeze(pressed) is True
    assert teleop_main._process_home_squeeze(pressed) is False
    assert robot.home_count == 1
    assert robot.reference_values == [[0.1] * 14]
    assert arm.wait_timeouts == [5.0]


def test_left_squeeze_saves_disarms_and_returns_home(monkeypatch):
    raw = FakeRawWriter()
    _configure_state(monkeypatch, raw)
    robot = FakeHomeRobot()
    arm = FakeHomeArmController()
    wrapper = SimpleNamespace(
        reset_left_wrist_ref=lambda: None,
        reset_right_wrist_ref=lambda: None,
    )
    monkeypatch.setattr(teleop_main, "robot", robot, raising=False)
    monkeypatch.setattr(teleop_main, "arm_ctrl", arm, raising=False)
    monkeypatch.setattr(teleop_main, "tv_wrapper", wrapper, raising=False)
    monkeypatch.setattr(teleop_main, "TELEOP_ACTIVE", True)
    monkeypatch.setattr(teleop_main, "LEFT_ARM_ACTIVE", True)
    monkeypatch.setattr(teleop_main, "RIGHT_ARM_ACTIVE", True)
    monkeypatch.setattr(teleop_main, "RECORD_RUNNING", True)
    monkeypatch.setattr(teleop_main, "CURRENT_RECORD_EPISODE_ID", 3)
    monkeypatch.setattr(teleop_main, "_home_left_squeeze_armed", True)

    pressed = SimpleNamespace(left_ctrl_squeeze=True, left_ctrl_squeezeValue=1.0)
    assert teleop_main._process_home_squeeze(pressed) is True
    assert raw.stopped == 1
    assert teleop_main.LAST_RECORD_EPISODE_ID == 3
    assert teleop_main.RECORD_RUNNING is False
    assert teleop_main.TELEOP_ACTIVE is False
    assert robot.home_count == 1


def test_discard_cannot_remove_episode_from_previous_process(monkeypatch, tmp_path):
    old_episode = tmp_path / "task" / "raw" / "episode_0004"
    old_episode.mkdir(parents=True)
    (old_episode / "manifest.json").write_text("{}", encoding="utf-8")
    raw = FakeRawWriter()
    _configure_state(monkeypatch, raw)

    teleop_main.RECORD_DISCARD = True
    teleop_main._process_record_requests()

    assert old_episode.exists()
    assert raw.discarded == []


def test_raw_only_start_save_and_discard(monkeypatch, tmp_path):
    task_dir = tmp_path / "task"
    raw = RawSessionWriter(task_dir, queue_size=16)
    _configure_state(monkeypatch, raw)

    teleop_main.RECORD_TOGGLE = True
    teleop_main._process_record_requests()
    assert teleop_main.RECORD_RUNNING is True
    assert teleop_main.CURRENT_RECORD_EPISODE_ID == 0
    assert raw.append_event("lowstate", {"sequence": 1})

    teleop_main.RECORD_TOGGLE = True
    teleop_main._process_record_requests()
    episode_dir = task_dir / "raw" / "episode_0000"
    manifest = json.loads((episode_dir / "manifest.json").read_text())
    assert manifest["complete"] is True
    assert manifest["metadata"]["recording_pipeline"] == (
        "legacy_control_raw_only_v1"
    )

    teleop_main.RECORD_DISCARD = True
    teleop_main._process_record_requests()
    assert not episode_dir.exists()
    raw.close()


def test_next_raw_episode_index_ignores_non_episode_names(tmp_path):
    raw = tmp_path / "raw"
    (raw / "episode_0002").mkdir(parents=True)
    (raw / "episode_bad").mkdir()
    assert teleop_main._next_raw_episode_index(tmp_path) == 3
