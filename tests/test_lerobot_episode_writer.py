import json
import unittest
from pathlib import Path

import numpy as np

from teleop.utils.lerobot_episode_writer import LeRobotEpisodeWriter


class FakeLeRobotDataset:
    created = []
    resumed = []

    def __init__(self, root, features, fps, repo_id, robot_type, use_videos, tolerance_s):
        self.root = Path(root)
        self.features = features
        self.fps = fps
        self.repo_id = repo_id
        self.robot_type = robot_type
        self.use_videos = use_videos
        self.tolerance_s = tolerance_s
        self.frames = []
        self.saved = 0
        self.finalized = 0
        self.num_episodes = 0

    @classmethod
    def create(cls, repo_id, fps, features, root, robot_type, use_videos, tolerance_s):
        dataset = cls(root, features, fps, repo_id, robot_type, use_videos, tolerance_s)
        cls.created.append(dataset)
        return dataset

    @classmethod
    def resume(cls, repo_id, root, tolerance_s):
        dataset = cls(root, {}, 0, repo_id, "resumed", True, tolerance_s)
        cls.resumed.append(dataset)
        return dataset

    def add_frame(self, frame):
        self.frames.append(frame)

    def save_episode(self):
        self.saved += 1
        self.num_episodes += 1

    def finalize(self):
        self.finalized += 1


def sample_item():
    return {
        "colors": {"color_0": np.zeros((4, 5, 3), dtype=np.uint8)},
        "depths": {"depth_0": np.ones((4, 5), dtype=np.uint16)},
        "states": {
            "left_arm": {"qpos": [0.1, 0.2], "qvel": [], "torque": []},
            "right_arm": {"qpos": [0.3, 0.4], "qvel": [], "torque": []},
            "body": {"qpos": [0.0]},
        },
        "actions": {
            "left_arm": {"qpos": [1.1, 1.2], "qvel": [], "torque": []},
            "right_arm": {"qpos": [1.3, 1.4], "qvel": [], "torque": []},
            "body": {"qpos": [0.5]},
        },
        "alignment": {
            "actual_timestamps_ns": {"control_cycle": 10, "robot_step_done": 20, "cameras": {"color_0": 15}},
            "camera_sync_skew_s": 0.0,
        },
    }


class LeRobotEpisodeWriterTest(unittest.TestCase):
    def setUp(self):
        import tempfile

        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmpdir.name)
        FakeLeRobotDataset.created.clear()
        FakeLeRobotDataset.resumed.clear()

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_add_item_creates_lerobot_dataset_and_adds_frame(self):
        writer = LeRobotEpisodeWriter(
            root=self.tmp_path / "dataset",
            repo_id="local/test",
            fps=30,
            task="pick cube",
            robot_type="topstar",
            dataset_cls=FakeLeRobotDataset,
        )

        self.assertTrue(writer.create_episode())
        writer.add_item(**sample_item())

        dataset = FakeLeRobotDataset.created[0]
        self.assertEqual(dataset.features["observation.state"]["shape"], (5,))
        self.assertEqual(dataset.features["action"]["shape"], (5,))
        self.assertEqual(dataset.features["observation.images.color_0"]["shape"], (4, 5, 3))
        self.assertEqual(dataset.features["observation.depths.depth_0"]["shape"], (4, 5, 3))
        self.assertEqual(dataset.frames[0]["task"], "pick cube")
        self.assertEqual(dataset.frames[0]["observation.state"].dtype, np.float32)
        self.assertEqual(dataset.frames[0]["action"].dtype, np.float32)

    def test_save_episode_writes_alignment_sidecar(self):
        writer = LeRobotEpisodeWriter(
            root=self.tmp_path / "dataset",
            repo_id="local/test",
            fps=30,
            task="pick cube",
            robot_type="topstar",
            dataset_cls=FakeLeRobotDataset,
        )

        writer.create_episode()
        writer.add_item(**sample_item())
        writer.save_episode()
        writer.close()

        dataset = FakeLeRobotDataset.created[0]
        self.assertEqual(dataset.saved, 1)
        self.assertEqual(dataset.finalized, 1)

        sidecar = self.tmp_path / "dataset" / "meta" / "tele_robot_alignment.jsonl"
        records = [json.loads(line) for line in sidecar.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(
            records,
            [
                {
                    "episode_index": 0,
                    "frame_index": 0,
                    "timestamp": 0.0,
                    "source_idx": 0,
                    "alignment": sample_item()["alignment"],
                }
            ],
        )


if __name__ == "__main__":
    unittest.main()
