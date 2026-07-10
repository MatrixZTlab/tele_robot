import json
import unittest
from pathlib import Path

import numpy as np

from teleop.utils.lerobot_episode_writer import (
    LeRobotEpisodeWriter,
    build_camera_key_plan,
)


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
    def create(cls, repo_id, fps, features, root, robot_type, use_videos, tolerance_s, vcodec="h264"):
        dataset = cls(root, features, fps, repo_id, robot_type, use_videos, tolerance_s)
        dataset.vcodec = vcodec
        cls.created.append(dataset)
        return dataset

    @classmethod
    def resume(cls, repo_id, root, tolerance_s):
        dataset = cls(root, {}, 0, repo_id, "resumed", True, tolerance_s)
        cls.resumed.append(dataset)
        return dataset

    def add_frame(self, frame):
        missing = set(self.features) - set(frame)
        if missing:
            raise ValueError(f"Missing features: {missing}")
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

    def test_missing_depth_reuses_previous_frame_value(self):
        writer = LeRobotEpisodeWriter(
            root=self.tmp_path / "dataset",
            repo_id="local/test",
            fps=30,
            task="pick cube",
            robot_type="topstar",
            dataset_cls=FakeLeRobotDataset,
        )

        first = sample_item()
        first["depths"]["depth_1"] = np.full((4, 5), 7, dtype=np.uint16)
        second = sample_item()

        writer.create_episode()
        writer.add_item(**first)
        writer.add_item(**second)

        dataset = FakeLeRobotDataset.created[0]
        self.assertIn("observation.depths.depth_1", dataset.frames[1])
        np.testing.assert_array_equal(
            dataset.frames[0]["observation.depths.depth_1"],
            dataset.frames[1]["observation.depths.depth_1"],
        )


class ColorAndCodecTest(unittest.TestCase):
    def setUp(self):
        import tempfile

        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmpdir.name)
        FakeLeRobotDataset.created.clear()
        FakeLeRobotDataset.resumed.clear()

    def tearDown(self):
        self.tmpdir.cleanup()

    def _writer(self, **kwargs):
        return LeRobotEpisodeWriter(
            root=self.tmp_path / "dataset",
            repo_id="local/test",
            fps=30,
            task="pick cube",
            robot_type="topstar",
            dataset_cls=FakeLeRobotDataset,
            **kwargs,
        )

    def test_bgr_input_is_flipped_to_rgb(self):
        # A distinct-channel image lets us see the flip: BGR [b,g,r] -> RGB [r,g,b].
        writer = self._writer(color_input_bgr=True)
        writer.create_episode()
        item = sample_item()
        bgr = np.zeros((2, 2, 3), dtype=np.uint8)
        bgr[..., 0] = 10  # B
        bgr[..., 1] = 20  # G
        bgr[..., 2] = 30  # R
        item["colors"] = {"color_0": bgr}
        item["depths"] = {}
        writer.add_item(**item)

        stored = FakeLeRobotDataset.created[0].frames[0]["observation.images.color_0"]
        # After flip, channel 0 should be the original R (30), channel 2 the B (10).
        self.assertEqual(int(stored[0, 0, 0]), 30)
        self.assertEqual(int(stored[0, 0, 2]), 10)

    def test_rgb_input_not_flipped(self):
        writer = self._writer(color_input_bgr=False)
        writer.create_episode()
        item = sample_item()
        rgb = np.zeros((2, 2, 3), dtype=np.uint8)
        rgb[..., 0] = 30  # R
        rgb[..., 2] = 10  # B
        item["colors"] = {"color_0": rgb}
        item["depths"] = {}
        writer.add_item(**item)

        stored = FakeLeRobotDataset.created[0].frames[0]["observation.images.color_0"]
        self.assertEqual(int(stored[0, 0, 0]), 30)
        self.assertEqual(int(stored[0, 0, 2]), 10)

    def test_default_vcodec_is_h264(self):
        writer = self._writer()
        writer.create_episode()
        writer.add_item(**sample_item())
        self.assertEqual(FakeLeRobotDataset.created[0].vcodec, "h264")


class BuildCameraKeyPlanTest(unittest.TestCase):
    def test_fixed_keys_by_config_order(self):
        config = {
            "head_camera": {"enable_zmq": True, "binocular": False, "enable_depth": False},
            "left_wrist_camera": {"enable_zmq": True, "binocular": False, "enable_depth": True},
            "right_wrist_camera": {"enable_zmq": True, "binocular": False, "enable_depth": True},
        }
        plan = build_camera_key_plan(config)
        self.assertEqual([e["source"] for e in plan], ["head", "left_wrist", "right_wrist"])
        self.assertEqual([e["color_key"] for e in plan], ["color_0", "color_1", "color_2"])
        # head has no depth; wrists do
        self.assertEqual([e["depth_key"] for e in plan], [None, "depth_1", "depth_2"])

    def test_disabled_camera_does_not_shift_others(self):
        # Left wrist disabled: right wrist must NOT slide into color_1.
        config = {
            "head_camera": {"enable_zmq": True, "binocular": False, "enable_depth": False},
            "left_wrist_camera": {"enable_zmq": False, "binocular": False, "enable_depth": True},
            "right_wrist_camera": {"enable_zmq": True, "binocular": False, "enable_depth": True},
        }
        plan = build_camera_key_plan(config)
        self.assertEqual([e["source"] for e in plan], ["head", "right_wrist"])
        # head=color_0, right_wrist=color_1 (fixed by enabled order)
        self.assertEqual([e["color_key"] for e in plan], ["color_0", "color_1"])

    def test_binocular_head_splits_into_two_keys(self):
        config = {
            "head_camera": {"enable_zmq": True, "binocular": True, "enable_depth": True},
        }
        plan = build_camera_key_plan(config)
        self.assertEqual([e["half"] for e in plan], [0, 1])
        self.assertEqual([e["color_key"] for e in plan], ["color_0", "color_1"])


class SchemaWarmupTest(unittest.TestCase):
    def setUp(self):
        import tempfile

        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmpdir.name)
        FakeLeRobotDataset.created.clear()
        FakeLeRobotDataset.resumed.clear()

    def tearDown(self):
        self.tmpdir.cleanup()

    def _writer(self, **kwargs):
        return LeRobotEpisodeWriter(
            root=self.tmp_path / "dataset",
            repo_id="local/test",
            fps=30,
            task="pick cube",
            robot_type="topstar",
            dataset_cls=FakeLeRobotDataset,
            **kwargs,
        )

    def test_late_camera_included_in_schema_and_backfilled(self):
        # color_1 only shows up on the 2nd frame. It must still be in the schema,
        # and the first (buffered) frame must be zero-filled for it.
        writer = self._writer(expected_color_keys=["color_0", "color_1"])
        writer.create_episode()

        first = sample_item()  # only color_0
        second = sample_item()
        second["colors"]["color_1"] = np.full((4, 5, 3), 9, dtype=np.uint8)

        writer.add_item(**first)
        # schema not committed yet: color_1 not seen, no dataset created
        self.assertEqual(len(FakeLeRobotDataset.created), 0)

        writer.add_item(**second)
        # now both cameras seen -> schema committed, both frames flushed
        self.assertEqual(len(FakeLeRobotDataset.created), 1)
        dataset = FakeLeRobotDataset.created[0]
        self.assertIn("observation.images.color_1", dataset.features)
        self.assertEqual(len(dataset.frames), 2)
        # frame 0 was buffered before color_1 existed -> zero-filled, not dropped
        np.testing.assert_array_equal(
            dataset.frames[0]["observation.images.color_1"],
            np.zeros((4, 5, 3), dtype=np.uint8),
        )

    def test_warmup_budget_commits_when_camera_never_arrives(self):
        # color_1 never arrives; after the warmup budget the schema commits
        # anyway (recording must not stall). Since color_1's shape was never
        # observed it cannot be fabricated, so it is EXCLUDED, not zero-filled.
        writer = self._writer(
            expected_color_keys=["color_0", "color_1"], schema_warmup_frames=3
        )
        writer.create_episode()
        for _ in range(3):
            writer.add_item(**sample_item())  # only color_0 ever present

        self.assertEqual(len(FakeLeRobotDataset.created), 1)
        dataset = FakeLeRobotDataset.created[0]
        self.assertNotIn("observation.images.color_1", dataset.features)
        self.assertIn("observation.images.color_0", dataset.features)
        self.assertEqual(len(dataset.frames), 3)

    def test_camera_arriving_after_commit_is_dropped_not_crashing(self):
        # A camera whose first frame lands AFTER the schema commits must be
        # silently dropped — never passed to add_frame (which rejects unknown
        # keys). Here only color_0 is expected, so a late color_1 is excluded.
        writer = self._writer(expected_color_keys=["color_0"])
        writer.create_episode()
        writer.add_item(**sample_item())  # commits schema with color_0 only
        self.assertEqual(len(FakeLeRobotDataset.created), 1)

        late = sample_item()
        late["colors"]["color_1"] = np.full((4, 5, 3), 3, dtype=np.uint8)
        writer.add_item(**late)  # must not raise

        dataset = FakeLeRobotDataset.created[0]
        self.assertNotIn("observation.images.color_1", dataset.frames[1])
        self.assertEqual(len(dataset.frames), 2)

    def test_short_episode_flushes_buffer_on_save(self):
        # Episode ends during warmup (before color_1 arrives): save_episode must
        # commit the schema from what we have and flush buffered frames.
        writer = self._writer(expected_color_keys=["color_0", "color_1"])
        writer.create_episode()
        writer.add_item(**sample_item())
        writer.save_episode()

        self.assertEqual(len(FakeLeRobotDataset.created), 1)
        dataset = FakeLeRobotDataset.created[0]
        self.assertEqual(len(dataset.frames), 1)
        self.assertEqual(dataset.saved, 1)


if __name__ == "__main__":
    unittest.main()
