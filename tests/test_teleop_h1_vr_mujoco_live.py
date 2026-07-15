import unittest
import threading
from types import SimpleNamespace

import numpy as np

from teleop.teleop_h1_vr_mujoco_live import (
    CameraBatch,
    FormalRecordingSession,
    H1MuJoCoLiveCore,
    LiveSafetyError,
    StateFreshnessMonitor,
    _build_parser,
    _seed_robot_command,
    DEFAULT_DRY_RUN_INITIAL_Q,
    build_lerobot_state_action,
    extract_camera_batch,
    limit_joint_speed,
)
from teleop.robot_control.vr_mujoco_relative_teleop import LMIKResult
from teleop.utils.lerobot_episode_writer import flatten_numeric_tree


class FakeSolver:
    def __init__(self):
        self.result_q = np.zeros(14)
        self.converged = True
        self.position_error = 0.001
        self.rotation_error = 0.001

    def forward_kinematics(self, _q):
        left = np.eye(4)
        right = np.eye(4)
        left[:3, 3] = [0.4, 0.2, 0.8]
        right[:3, 3] = [0.4, -0.2, 0.8]
        return left, right

    def solve(self, *_args, **kwargs):
        self.last_solve_kwargs = kwargs
        return LMIKResult(
            arm_q=self.result_q,
            converged=self.converged,
            iterations=2,
            translation_error_norm=self.position_error,
            rotation_error_norm=self.rotation_error,
        )


class LiveHelpersTest(unittest.TestCase):
    def test_cli_defaults_to_dry_run(self):
        args = _build_parser().parse_args([])
        self.assertFalse(args.live)
        np.testing.assert_allclose(
            args.dry_run_initial_q, DEFAULT_DRY_RUN_INITIAL_Q
        )

    def test_joint_speed_limit_is_per_joint(self):
        target = np.linspace(-1.0, 1.0, 14)
        result = limit_joint_speed(np.zeros(14), target, 0.5, 0.1)
        self.assertLessEqual(float(np.max(np.abs(result))), 0.05 + 1e-12)

    def test_state_freshness_requires_replacement(self):
        monitor = StateFreshnessMonitor(0.5)
        state = object()
        self.assertTrue(monitor.observe(state, now=1.0))
        self.assertTrue(monitor.observe(state, now=1.4))
        self.assertFalse(monitor.observe(state, now=1.6))
        self.assertTrue(monitor.observe(object(), now=1.7))

    def test_seed_robot_command_holds_measured_head_and_torso(self):
        controller = SimpleNamespace(
            publish_lock=threading.Lock(),
            get_head_q=lambda: np.array([0.2, -0.1]),
        )
        motors = [SimpleNamespace(q=0.3), SimpleNamespace(q=0.4)]
        state = SimpleNamespace(motor_state=motors)
        arm_q = np.linspace(-0.5, 0.5, 14)

        _seed_robot_command(controller, arm_q, state)

        np.testing.assert_allclose(controller.q_target, arm_q)
        np.testing.assert_allclose(controller.last_published_q, arm_q)
        np.testing.assert_allclose(controller.head_target, [0.2, -0.1])
        np.testing.assert_allclose(controller.torso_target, [0.3, 0.4])

    def test_recording_payload_matches_existing_28_value_schema(self):
        measured = np.arange(14, dtype=float)
        command = measured + 0.5
        states, actions = build_lerobot_state_action(measured, command)

        state_values, _ = flatten_numeric_tree(states)
        action_values, _ = flatten_numeric_tree(actions)

        self.assertEqual(len(state_values), 28)
        self.assertEqual(len(action_values), 28)
        np.testing.assert_allclose(state_values[:14], measured)
        np.testing.assert_allclose(state_values[14:], measured)
        np.testing.assert_allclose(action_values[:14], command)
        np.testing.assert_allclose(action_values[14:], command)

    def test_camera_batch_uses_fixed_keys_and_timestamps(self):
        config = {
            "head_camera": {"image_shape": (2, 4)},
            "left_wrist_camera": {"image_shape": (2, 2)},
        }
        plan = [
            {
                "cam": "head_camera",
                "source": "head",
                "half": 0,
                "color_key": "color_0",
                "depth_key": None,
            },
            {
                "cam": "left_wrist_camera",
                "source": "left_wrist",
                "half": None,
                "color_key": "color_1",
                "depth_key": "depth_1",
            },
        ]
        images = {
            "head": SimpleNamespace(
                bgr=np.zeros((2, 4, 3), dtype=np.uint8),
                depth=None,
                timestamp_ns=100,
            ),
            "left_wrist": SimpleNamespace(
                bgr=np.ones((2, 2, 3), dtype=np.uint8),
                depth=np.ones((2, 2), dtype=np.uint16),
                timestamp_ns=120,
            ),
        }

        batch = extract_camera_batch(config, plan, images)

        self.assertEqual(batch.colors["color_0"].shape, (2, 2, 3))
        self.assertEqual(batch.colors["color_1"].shape, (2, 2, 3))
        self.assertEqual(batch.depths["depth_1"].shape, (2, 2))
        self.assertEqual(batch.timestamps_ns, {"color_0": 100, "color_1": 120})


class LiveCoreTest(unittest.TestCase):
    def setUp(self):
        self.solver = FakeSolver()
        self.core = H1MuJoCoLiveCore(
            self.solver,
            max_joint_speed_rad_s=0.5,
            max_solver_jump_rad=0.2,
        )
        self.pose = np.eye(4)
        self.core.arm(np.zeros(14), self.pose, self.pose)

    def test_arm_captures_reference_without_command_motion(self):
        result = self.core.step(np.zeros(14), self.pose, self.pose, 0.05)
        np.testing.assert_allclose(result.command_q, np.zeros(14))

    def test_speed_limit_applies_to_converged_ik(self):
        self.solver.result_q[:] = 0.1
        result = self.core.step(np.zeros(14), self.pose, self.pose, 0.05)
        np.testing.assert_allclose(result.command_q, np.full(14, 0.025))

    def test_nonconvergence_disarms_at_call_site(self):
        self.solver.converged = False
        self.solver.position_error = 0.08
        with self.assertRaisesRegex(LiveSafetyError, "safety envelope"):
            self.core.step(np.zeros(14), self.pose, self.pose, 0.05)

    def test_bounded_nonconvergence_is_accepted_as_approximate(self):
        self.solver.converged = False
        self.solver.position_error = 0.02
        self.solver.rotation_error = 0.03
        result = self.core.step(np.zeros(14), self.pose, self.pose, 0.05)
        self.assertTrue(result.accepted_approximately)

    def test_large_solver_branch_jump_is_rejected(self):
        self.solver.result_q[3] = 0.21
        with self.assertRaisesRegex(LiveSafetyError, "joint jump"):
            self.core.step(np.zeros(14), self.pose, self.pose, 0.05)

    def test_dry_run_can_hold_last_command_on_ik_safety_event(self):
        core = H1MuJoCoLiveCore(
            self.solver,
            max_joint_speed_rad_s=0.5,
            max_solver_jump_rad=0.2,
            hold_on_ik_safety=True,
        )
        core.arm(np.zeros(14), self.pose, self.pose)
        self.solver.result_q[3] = 0.21

        result = core.step(np.zeros(14), self.pose, self.pose, 0.05)

        self.assertTrue(result.held_for_ik_safety)
        np.testing.assert_allclose(result.command_q, np.zeros(14))

    def test_speed_lag_is_not_mistaken_for_an_ik_branch_jump(self):
        self.solver.result_q[:] = 0.15
        first = self.core.step(np.zeros(14), self.pose, self.pose, 0.05)
        np.testing.assert_allclose(first.command_q, np.full(14, 0.025))

        # The new target is close to the previous raw IK solution but still
        # far ahead of the speed-limited command. This must remain valid.
        self.solver.result_q[:] = 0.30
        second = self.core.step(np.zeros(14), self.pose, self.pose, 0.05)
        np.testing.assert_allclose(second.command_q, np.full(14, 0.05))

    def test_large_robot_tracking_error_is_rejected(self):
        measured = np.zeros(14)
        measured[4] = 0.36
        with self.assertRaisesRegex(LiveSafetyError, "tracking error"):
            self.core.step(measured, self.pose, self.pose, 0.05)

    def test_disarmed_core_rejects_step(self):
        self.core.disarm()
        with self.assertRaisesRegex(LiveSafetyError, "not armed"):
            self.core.step(np.zeros(14), self.pose, self.pose, 0.05)

    def test_regularization_params_reach_solver(self):
        # The null-space regularization is the fix for elbow-flip IK jumps, so
        # current_q_weight/home_qpos/home_weight MUST actually reach solve().
        home = np.linspace(-0.3, 0.3, 14)
        core = H1MuJoCoLiveCore(
            FakeSolver(),
            current_q_weight=0.05,
            home_qpos=home,
            home_weight=0.02,
        )
        core.arm(np.zeros(14), self.pose, self.pose)
        core.step(np.zeros(14), self.pose, self.pose, 0.05)

        kwargs = core.solver.last_solve_kwargs
        self.assertEqual(kwargs["current_q_weight"], 0.05)
        self.assertEqual(kwargs["home_weight"], 0.02)
        np.testing.assert_allclose(kwargs["home_qpos"], home)

    def test_default_current_q_weight_is_nonzero(self):
        # Guards against regressing to the vr_teleop-port bug where the default
        # was 0.0 and the redundant arm flipped IK branches.
        core = H1MuJoCoLiveCore(FakeSolver())
        self.assertGreater(core.current_q_weight, 0.0)
        self.assertIsNone(core.home_qpos)
        self.assertEqual(core.home_weight, 0.0)


class FakeEpisodeWriter:
    def __init__(self):
        self.frames = []
        self.saved = 0
        self.closed = False

    def create_episode(self):
        return True

    def add_item(self, **kwargs):
        self.frames.append(kwargs)

    def save_episode(self):
        self.saved += 1

    def close(self):
        self.closed = True


class FakeRecorder:
    instances = []

    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.points = []
        self._servo_recording = False
        self.finished = False
        FakeRecorder.instances.append(self)

    def start_program(self, *_args, **_kwargs):
        pass

    def start_servo_recording(self):
        self._servo_recording = True

    def add_ServoJ(self, joint_pose, **kwargs):
        self.points.append((joint_pose, kwargs))

    def stop_servo_recording(self, smooth_window_size):
        self._servo_recording = False

    def finish(self):
        self.finished = True


class FormalRecordingSessionTest(unittest.TestCase):
    def setUp(self):
        FakeRecorder.instances = []
        self.image = SimpleNamespace(
            bgr=np.zeros((2, 2, 3), dtype=np.uint8),
            depth=None,
            timestamp_ns=100,
        )
        self.client = SimpleNamespace(
            get_head_frame=lambda: self.image,
            get_left_wrist_frame=lambda: self.image,
            get_right_wrist_frame=lambda: self.image,
        )
        self.writer = FakeEpisodeWriter()
        self.session = FormalRecordingSession(
            image_client=self.client,
            camera_config={},
            camera_key_plan=[],
            episode_writer=self.writer,
            recorder_cls=FakeRecorder,
            output_root=SimpleNamespace(__str__=lambda _self: "/tmp/fake"),
            task_name="test",
            task_description="test",
            frequency=20,
            sync_tolerance_s=1e-4,
            camera_sync_tolerance_s=0.02,
        )

    def test_missing_camera_prevents_episode_start(self):
        self.client.get_right_wrist_frame = lambda: None
        with self.assertRaisesRegex(RuntimeError, "right_wrist"):
            self.session.start()
        self.assertFalse(self.session.active)

    def test_frame_writes_lerobot_and_legacy_action(self):
        self.session.start()
        ik = LMIKResult(
            arm_q=np.full(14, 0.2),
            converged=True,
            iterations=3,
            translation_error_norm=0.001,
            rotation_error_norm=0.002,
        )
        batch = CameraBatch({}, {}, {"color_0": 100, "color_1": 120})

        self.session.add_frame(
            camera_batch=batch,
            measured_q=np.full(14, 0.1),
            command_q=np.full(14, 0.15),
            raw_ik_q=np.full(14, 0.2),
            ik_result=ik,
            accepted_approximately=False,
            control_cycle_timestamp_ns=200,
            robot_step_done_timestamp_ns=300,
        )
        self.session.stop()

        self.assertEqual(len(self.writer.frames), 1)
        self.assertEqual(self.writer.saved, 1)
        recorder = FakeRecorder.instances[0]
        self.assertEqual(len(recorder.points), 1)
        np.testing.assert_allclose(recorder.points[0][0], np.degrees(0.15))
        self.assertTrue(recorder.finished)

if __name__ == "__main__":
    unittest.main()
