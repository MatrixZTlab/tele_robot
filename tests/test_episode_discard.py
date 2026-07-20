import time
from threading import Event, Thread

from teleop.utils.episode_writer import EpisodeWriter


def _wait_until_ready(writer, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if writer.is_ready():
            return
        time.sleep(0.01)
    raise AssertionError("EpisodeWriter did not become ready")


def test_discard_active_episode_removes_directory(tmp_path):
    writer = EpisodeWriter(tmp_path / "task", rerun_log=False)
    assert writer.create_episode()
    episode_dir = tmp_path / "task" / "episode_0000"
    writer.add_item(colors={}, states={"q": [0.1]}, actions={"q": [0.2]})

    assert writer.discard_episode() == str(episode_dir)
    _wait_until_ready(writer)

    assert not episode_dir.exists()
    writer.close()


def test_discard_saved_episode_removes_directory(tmp_path):
    writer = EpisodeWriter(tmp_path / "task", rerun_log=False)
    assert writer.create_episode()
    episode_dir = tmp_path / "task" / "episode_0000"
    writer.add_item(colors={}, states={"q": [0.1]}, actions={"q": [0.2]})
    writer.save_episode()
    _wait_until_ready(writer)
    assert episode_dir.exists()

    assert writer.discard_episode() == str(episode_dir)
    _wait_until_ready(writer)

    assert not episode_dir.exists()
    writer.close()


def test_discard_targets_requested_episode_only(tmp_path):
    task_dir = tmp_path / "task"
    old_episode = task_dir / "episode_0004"
    old_episode.mkdir(parents=True)
    (old_episode / "data.json").write_text("old", encoding="utf-8")

    writer = EpisodeWriter(task_dir, rerun_log=False)
    assert writer.create_episode()
    new_episode = task_dir / "episode_0005"

    assert writer.discard_episode(5) == str(new_episode)
    _wait_until_ready(writer)

    assert old_episode.exists()
    assert not new_episode.exists()
    writer.close()


def test_discard_during_save_cannot_delete_next_episode(tmp_path):
    writer = EpisodeWriter(tmp_path / "task", rerun_log=False)
    assert writer.create_episode()
    first_episode = tmp_path / "task" / "episode_0000"
    writer.add_item(colors={}, states={"q": [0.1]}, actions={"q": [0.2]})

    save_started = Event()
    allow_save = Event()
    original_save = writer._save_episode

    def delayed_save():
        save_started.set()
        assert allow_save.wait(timeout=2.0)
        original_save()

    writer._save_episode = delayed_save
    writer.save_episode()
    assert save_started.wait(timeout=2.0)
    assert writer.discard_episode() == str(first_episode)
    allow_save.set()
    _wait_until_ready(writer)

    assert not first_episode.exists()
    assert writer.create_episode()
    second_episode = tmp_path / "task" / "episode_0001"
    assert second_episode.exists()
    writer.discard_episode()
    _wait_until_ready(writer)
    writer.close()


def test_close_during_discard_does_not_recreate_episode(tmp_path):
    writer = EpisodeWriter(tmp_path / "task", rerun_log=False)
    assert writer.create_episode()
    episode_dir = tmp_path / "task" / "episode_0000"
    writer.add_item(colors={}, states={"q": [0.1]}, actions={"q": [0.2]})

    discard_started = Event()
    allow_discard = Event()
    original_discard = writer._discard_episode

    def delayed_discard(path):
        discard_started.set()
        assert allow_discard.wait(timeout=2.0)
        original_discard(path)

    writer._discard_episode = delayed_discard
    assert writer.discard_episode() == str(episode_dir)
    assert discard_started.wait(timeout=2.0)

    close_thread = Thread(target=writer.close)
    close_thread.start()
    time.sleep(0.05)
    assert close_thread.is_alive()
    allow_discard.set()
    close_thread.join(timeout=2.0)

    assert not close_thread.is_alive()
    assert not episode_dir.exists()
