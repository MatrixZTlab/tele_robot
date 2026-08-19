import unittest
import threading
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

import teleop.teleop_h1_vr_mujoco_live as live_module
from teleop.teleop_h1_vr_mujoco_live import (
    CameraBatch,
    FormalRecordingSession,
    H1MuJoCoLiveCore,
    LiveSafetyError,
    StateFreshnessMonitor,
    _build_parser,
    _send_robot_hold,
    _seed_robot_command,
    _validate_args,
    _validate_robot_arm_state,
    _validate_ros_command_graph,
    DEFAULT_DRY_RUN_INITIAL_Q,
    build_lerobot_state_action,
    extract_camera_batch,
    limit_joint_speed,
    mapped_wrist_poses,
    monotonic_timestamp_is_fresh,
)
from teleop.robot_control.vr_mujoco_relative_teleop import (
    H1_HARD_ARM_LOWER,
    H1_HARD_ARM_UPPER,
    H1MuJoCoLMIK,
    LMIKResult,
)
from teleop.robot_control.topstar_h1.joint_convention import (
    H1_ARM_COORDINATE_CONVENTION,
    H1_MODEL_REVISION,
)
from teleop.utils.lerobot_episode_writer import flatten_numeric_tree


class FakeSolver:
    def __init__(self):
        self.result_q = np.zeros(14)
        self.converged = True
        self.position_error = 0.001
        self.rotation_error = 0.001
        self.lower = np.full(14, -1.0)
        self.upper = np.full(14, 1.0)

    def forward_kinematics(self, _q):
        self.last_forward_q = np.asarray(_q).copy()
        left = np.eye(4)
        right = np.eye(4)
        left[:3, 3] = [0.4, 0.2, 0.8]
        right[:3, 3] = [0.4, -0.2, 0.8]
        return left, right

    def solve(self, *_args, **kwargs):
        self.last_solve_kwargs = kwargs
        self.last_left_target = np.asarray(_args[0]).copy()
        self.last_right_target = np.asarray(_args[1]).copy()
        self.last_solve_q = np.asarray(_args[2]).copy()
        return LMIKResult(
            arm_q=self.result_q,
            converged=self.converged,
            iterations=2,
            translation_error_norm=self.position_error,
            rotation_error_norm=self.rotation_error,
        )

    @property
    def arm_joint_limits(self):
        return self.lower.copy(), self.upper.copy()

    def arm_q_within_limits(self, q, *, tolerance=0.0):
        value = np.asarray(q)
        return bool(
            np.all(value >= self.lower - tolerance)
            and np.all(value <= self.upper + tolerance)
        )

    def gravity_compensation(self, q):
        return np.asarray(q, dtype=float) + 0.25


class LiveHelpersTest(unittest.TestCase):
    def test_cli_defaults_to_dry_run(self):
        args = _build_parser().parse_args([])
        self.assertFalse(args.live)
        self.assertFalse(args.allow_hard_limit_start)
        self.assertFalse(args.disable_tracking_error_check)
        self.assertFalse(args.disable_ik_safety_checks)
        self.assertFalse(args.legacy_stale_state_behavior)
        self.assertEqual(args.legacy_stale_state_hard_timeout, 2.0)
        self.assertIsNone(args.record_cameras)
        self.assertEqual(args.orientation_control, "vr")
        self.assertIsNone(args.controller_mapping)
        self.assertEqual(args.arm_scale, 0.7)
        self.assertEqual(args.state_timeout, 0.25)
        self.assertEqual(args.pico_timeout, 0.25)
        self.assertEqual(args.current_q_weight, 0.0001)
        np.testing.assert_allclose(
            args.dry_run_initial_q, DEFAULT_DRY_RUN_INITIAL_Q
        )

    def test_live_requires_explicit_mapping_and_rejects_viewer(self):
        parser = _build_parser()
        with self.assertRaisesRegex(ValueError, "controller-mapping"):
            _validate_args(parser.parse_args(["--live"]))
        with self.assertRaisesRegex(ValueError, "disabled with --live"):
            _validate_args(
                parser.parse_args(
                    [
                        "--live",
                        "--controller-mapping",
                        "mirrored",
                        "--mujoco-viewer",
                    ]
                )
            )
        _validate_args(
            parser.parse_args(
                ["--live", "--controller-mapping", "mirrored"]
            )
        )
        with self.assertRaisesRegex(ValueError, "hard-timeout"):
            _validate_args(
                parser.parse_args(
                    [
                        "--legacy-stale-state-behavior",
                        "--state-timeout",
                        "0.5",
                        "--legacy-stale-state-hard-timeout",
                        "0.5",
                    ]
                )
            )
        args = parser.parse_args(["--record-cameras", "head"])
        self.assertEqual(args.record_cameras, ["head"])
        with self.assertRaisesRegex(ValueError, "raw-only"):
            _validate_args(
                parser.parse_args(["--record", "--no-raw-record"])
            )
        args = parser.parse_args(["--orientation-control", "locked"])
        self.assertEqual(args.orientation_control, "locked")
        _validate_args(
            parser.parse_args(
                [
                    "--live",
                    "--controller-mapping",
                    "mirrored",
                    "--disable-tracking-error-check",
                ]
            )
        )

    def test_monotonic_input_freshness_rejects_missing_stale_and_future(self):
        now_ns = 2_000_000_000
        self.assertTrue(
            monotonic_timestamp_is_fresh(
                1_800_000_000, 0.25, now_ns=now_ns
            )
        )
        self.assertFalse(
            monotonic_timestamp_is_fresh(0, 0.25, now_ns=now_ns)
        )
        self.assertFalse(
            monotonic_timestamp_is_fresh(
                1_700_000_000, 0.25, now_ns=now_ns
            )
        )
        self.assertFalse(
            monotonic_timestamp_is_fresh(
                2_020_000_000, 0.25, now_ns=now_ns
            )
        )

    def test_controller_mapping_is_explicit(self):
        left = np.eye(4)
        right = np.eye(4)
        left[0, 3] = 1.0
        right[0, 3] = 2.0
        tele_data = SimpleNamespace(
            left_wrist_pose=left, right_wrist_pose=right
        )

        same_left, same_right = mapped_wrist_poses(tele_data, "same-side")
        mirror_left, mirror_right = mapped_wrist_poses(
            tele_data, "mirrored"
        )

        self.assertIs(same_left, left)
        self.assertIs(same_right, right)
        self.assertIs(mirror_left, right)
        self.assertIs(mirror_right, left)

    def test_measured_state_limit_and_gravity_hold_helpers(self):
        solver = FakeSolver()
        q, dq = _validate_robot_arm_state(
            solver, np.zeros(14), np.zeros(14)
        )
        np.testing.assert_allclose(q, 0.0)
        np.testing.assert_allclose(dq, 0.0)
        outside = np.zeros(14)
        outside[3] = 1.031
        with self.assertRaisesRegex(LiveSafetyError, r"q\[3\]"):
            _validate_robot_arm_state(solver, outside, np.zeros(14))

        sent = {}
        controller = SimpleNamespace(
            servo_dual_arm=lambda command, tau: sent.update(
                command=np.asarray(command), tau=np.asarray(tau)
            )
        )
        tau = _send_robot_hold(controller, solver, np.zeros(14))
        np.testing.assert_allclose(sent["command"], 0.0)
        np.testing.assert_allclose(sent["tau"], 0.25)
        np.testing.assert_allclose(tau, 0.25)

    def test_ros_graph_rejects_duplicate_lowcmd_publisher(self):
        valid = SimpleNamespace(
            _ros_node=SimpleNamespace(
                count_publishers=lambda _topic: 1,
                count_subscribers=lambda _topic: 1,
            )
        )
        self.assertEqual(
            _validate_ros_command_graph(valid, settle_timeout_s=0.0),
            (1, 1),
        )

        duplicate = SimpleNamespace(
            _ros_node=SimpleNamespace(
                count_publishers=lambda _topic: 2,
                count_subscribers=lambda _topic: 1,
            )
        )
        with self.assertRaisesRegex(LiveSafetyError, "exactly one"):
            _validate_ros_command_graph(duplicate, settle_timeout_s=0.0)

    def test_state_timeout_does_not_publish_zero_or_hold_command(self):
        class FakeController:
            def __init__(self):
                self._arm_active = False
                self.commands = []
                self.stopped = False

            def servo_dual_arm(self, q, tau):
                self.commands.append((q, tau))

            def stop(self):
                self.stopped = True

        controller = FakeController()
        with (
            mock.patch.object(
                live_module.H1MuJoCoLMIK,
                "from_h1_assets",
                return_value=FakeSolver(),
            ),
            mock.patch.object(
                live_module,
                "_start_h1_controller",
                return_value=controller,
            ),
            mock.patch.object(
                live_module,
                "_wait_for_robot_state",
                side_effect=RuntimeError("no state"),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "no state"):
                live_module.main(
                    ["--live", "--controller-mapping", "mirrored"]
                )

        self.assertEqual(controller.commands, [])
        self.assertTrue(controller.stopped)

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

    def test_mirrored_mode_flips_only_lateral_translation(self):
        solver = FakeSolver()
        core = H1MuJoCoLiveCore(
            solver,
            mirror_lateral_translation=True,
            max_solver_jump_rad=0.2,
        )
        core.arm(np.zeros(14), self.pose, self.pose)
        moved = np.eye(4)
        moved[:3, 3] = [0.1, 0.2, 0.3]

        core.step(np.zeros(14), moved, moved, 0.05)

        # H1 basis maps XR [x, y, z] to robot [z, x, y]. Face-to-face
        # mirroring reverses only robot lateral Y, leaving forward X and up Z.
        expected_delta = np.array([0.3, -0.1, 0.2])
        np.testing.assert_allclose(
            solver.last_left_target[:3, 3],
            np.array([0.4, 0.2, 0.8]) + expected_delta,
        )
        np.testing.assert_allclose(
            solver.last_right_target[:3, 3],
            np.array([0.4, -0.2, 0.8]) + expected_delta,
        )

    def test_locked_orientation_ignores_both_wrist_rotations(self):
        solver = FakeSolver()
        core = H1MuJoCoLiveCore(
            solver,
            orientation_control="locked",
            max_solver_jump_rad=0.2,
        )
        core.arm(np.zeros(14), self.pose, self.pose)
        moved = np.eye(4)
        moved[:3, :3] = np.array(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        )
        moved[0, 3] = 0.1

        core.step(np.zeros(14), moved, moved, 0.05)

        np.testing.assert_allclose(solver.last_left_target[:3, :3], np.eye(3))
        np.testing.assert_allclose(solver.last_right_target[:3, :3], np.eye(3))
        self.assertFalse(np.allclose(solver.last_left_target[:3, 3], [0.4, 0.2, 0.8]))

    def test_right_a_style_box_lock_rejects_differential_hand_motion(self):
        locked, width = self.core.toggle_dual_object_lock(
            np.zeros(14), self.pose, self.pose, now_s=1.0
        )
        self.assertTrue(locked)
        self.assertTrue(self.core.dual_object_locked)
        self.assertAlmostEqual(width, 0.4)

        left_moved = np.eye(4)
        right_moved = np.eye(4)
        left_moved[0, 3] = 0.10
        right_moved[0, 3] = -0.10
        self.core.step(np.zeros(14), left_moved, right_moved, 0.05)

        # Opposite controller motion is rejected while locked: the measured
        # grasp width and both robot-side endpoints remain fixed.
        np.testing.assert_allclose(
            self.solver.last_left_target[:3, 3], [0.4, 0.2, 0.8]
        )
        np.testing.assert_allclose(
            self.solver.last_right_target[:3, 3], [0.4, -0.2, 0.8]
        )

        locked, width = self.core.toggle_dual_object_lock(
            np.zeros(14), left_moved, right_moved, now_s=1.1
        )
        self.assertFalse(locked)
        self.assertIsNone(width)
        self.assertFalse(self.core.dual_object_locked)

    def test_hard_limit_start_holds_exact_measured_pose(self):
        solver = FakeSolver()
        solver.lower = H1_HARD_ARM_LOWER.copy()
        solver.upper = H1_HARD_ARM_UPPER.copy()
        core = H1MuJoCoLiveCore(
            solver,
            max_joint_speed_rad_s=0.5,
            max_solver_jump_rad=0.2,
        )
        current = np.zeros(14)
        current[4] = H1_HARD_ARM_LOWER[4]
        current[6] = H1_HARD_ARM_UPPER[6]
        solver.result_q = current.copy()
        core.arm(current, self.pose, self.pose)

        result = core.step(current, self.pose, self.pose, 0.05)

        np.testing.assert_allclose(solver.last_forward_q, current)
        np.testing.assert_allclose(solver.last_solve_q, current)
        np.testing.assert_allclose(result.command_q, current)

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

    def test_solver_output_outside_joint_limits_is_rejected(self):
        core = H1MuJoCoLiveCore(
            self.solver,
            max_solver_jump_rad=2.0,
        )
        core.arm(np.zeros(14), self.pose, self.pose)
        self.solver.result_q[3] = 1.01

        with self.assertRaisesRegex(LiveSafetyError, "joint-limit"):
            core.step(np.zeros(14), self.pose, self.pose, 0.05)

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

    def test_dry_run_can_explicitly_accept_nonconverged_ik(self):
        core = H1MuJoCoLiveCore(
            self.solver,
            max_joint_speed_rad_s=30.0,
            max_solver_jump_rad=0.01,
            ik_safety_checks_enabled=False,
        )
        core.arm(np.zeros(14), self.pose, self.pose)
        self.solver.converged = False
        self.solver.position_error = 0.5
        self.solver.rotation_error = 0.5
        self.solver.result_q[:] = 0.5

        result = core.step(np.zeros(14), self.pose, self.pose, 0.05)

        self.assertFalse(result.held_for_ik_safety)
        self.assertTrue(result.accepted_approximately)
        np.testing.assert_allclose(result.command_q, np.full(14, 0.5))

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

    def test_tracking_error_check_can_be_explicitly_disabled(self):
        core = H1MuJoCoLiveCore(
            self.solver,
            max_joint_speed_rad_s=0.5,
            max_solver_jump_rad=0.2,
            tracking_error_check_enabled=False,
        )
        core.arm(np.zeros(14), self.pose, self.pose)
        measured = np.zeros(14)
        measured[4] = 0.36

        result = core.step(measured, self.pose, self.pose, 0.05)

        self.assertEqual(result.command_q.shape, (14,))
        self.assertTrue(np.all(np.isfinite(result.command_q)))

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
        self.assertEqual(core.current_q_weight, 0.0001)
        self.assertIsNone(core.home_qpos)
        self.assertEqual(core.home_weight, 0.0)


try:
    import mujoco  # type: ignore
except ImportError:
    mujoco = None


@unittest.skipUnless(mujoco is not None, "MuJoCo not installed")
class H1HardLimitHoldIntegrationTest(unittest.TestCase):
    def test_first_frame_holds_two_wrist_joints_at_measured_pose(self):
        solver = H1MuJoCoLMIK.from_h1_assets(
            Path(__file__).resolve().parents[1], arm_limit_mode="hard"
        )
        core = H1MuJoCoLiveCore(
            solver,
            max_joint_speed_rad_s=0.15,
            max_solver_jump_rad=0.20,
            max_tracking_error_rad=0.20,
        )
        current = np.zeros(14)
        current[4] = H1_HARD_ARM_LOWER[4]
        current[6] = H1_HARD_ARM_UPPER[6]
        wrist_pose = np.eye(4)

        core.arm(current, wrist_pose, wrist_pose)
        result = core.step(current, wrist_pose, wrist_pose, 0.05)

        self.assertTrue(result.ik.converged)
        np.testing.assert_allclose(result.raw_ik_q, current, atol=1e-10)
        np.testing.assert_allclose(result.command_q, current, atol=1e-10)


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


class FakeRawWriter:
    def __init__(self):
        self.started = []
        self.events = []
        self.stopped = 0
        self.closed = False

    def start_episode(self, episode_index, metadata=None):
        self.started.append((episode_index, metadata))

    def append_event(self, stream, event):
        self.events.append((stream, event))

    def stop_episode(self):
        self.stopped += 1

    def close(self):
        self.closed = True


class FormalRecordingSessionTest(unittest.TestCase):
    def setUp(self):
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
        self.camera_config = {
            "head_camera": {"enable_zmq": True},
            "left_wrist_camera": {"enable_zmq": True},
            "right_wrist_camera": {"enable_zmq": True},
        }
        self.camera_key_plan = [
            {"source": "head"},
            {"source": "left_wrist"},
            {"source": "right_wrist"},
        ]
        self.session = FormalRecordingSession(
            image_client=self.client,
            camera_config=self.camera_config,
            camera_key_plan=self.camera_key_plan,
            legacy_episode_writer=self.writer,
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

    def test_head_only_session_does_not_read_disabled_wrist_cameras(self):
        self.client.get_left_wrist_frame = mock.Mock(
            side_effect=AssertionError("left wrist must not be read")
        )
        self.client.get_right_wrist_frame = mock.Mock(
            side_effect=AssertionError("right wrist must not be read")
        )
        session = FormalRecordingSession(
            image_client=self.client,
            camera_config={"head_camera": {"enable_zmq": True}},
            camera_key_plan=[{"source": "head"}],
            legacy_episode_writer=self.writer,
            output_root=SimpleNamespace(__str__=lambda _self: "/tmp/fake"),
            task_name="head_only",
            task_description="head only",
            frequency=20,
            sync_tolerance_s=1e-4,
            camera_sync_tolerance_s=0.02,
        )

        session.start()

        self.assertTrue(session.active)
        self.client.get_left_wrist_frame.assert_not_called()
        self.client.get_right_wrist_frame.assert_not_called()

    def test_frame_queues_legacy_episode_without_trajectory_json(self):
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

    def test_raw_only_session_does_not_queue_legacy_images(self):
        raw = FakeRawWriter()
        self.image.jpg = b"jpeg"
        with tempfile.TemporaryDirectory() as tmp:
            session = FormalRecordingSession(
                image_client=self.client,
                camera_config=self.camera_config,
                camera_key_plan=[
                    {
                        "source": "head",
                        "color_key": "color_0",
                    },
                    {
                        "source": "left_wrist",
                        "color_key": "color_1",
                    },
                    {
                        "source": "right_wrist",
                        "color_key": "color_2",
                    },
                ],
                legacy_episode_writer=None,
                output_root=Path(tmp),
                task_name="raw_only",
                task_description="raw only",
                frequency=20,
                sync_tolerance_s=1e-4,
                camera_sync_tolerance_s=0.02,
                raw_writer=raw,
            )
            session.start()
            batch = session.capture_camera_batch()
            self.assertEqual(batch.colors, {})
            self.assertEqual(batch.depths, {})
            self.assertEqual(
                batch.timestamps_ns,
                {"color_0": 100, "color_1": 100, "color_2": 100},
            )
            session.add_frame(
                camera_batch=batch,
                measured_q=np.full(14, 0.1),
                command_q=np.full(14, 0.15),
                raw_ik_q=np.full(14, 0.2),
                ik_result=LMIKResult(
                    arm_q=np.full(14, 0.2),
                    converged=True,
                    iterations=3,
                    translation_error_norm=0.001,
                    rotation_error_norm=0.002,
                ),
                accepted_approximately=False,
                control_cycle_timestamp_ns=200,
                robot_step_done_timestamp_ns=300,
            )
            session.stop()

        self.assertEqual(raw.started[0][0], 0)
        self.assertEqual(raw.stopped, 1)
        self.assertEqual([name for name, _ in raw.events], ["control"])

if __name__ == "__main__":
    unittest.main()
