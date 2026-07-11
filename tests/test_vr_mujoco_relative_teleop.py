import importlib
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from teleop.robot_control.vr_mujoco_relative_teleop import (
    H1_LEFT_ARM_JOINT_NAMES,
    H1_RIGHT_ARM_JOINT_NAMES,
    H1_XR_TO_ROBOT_ROTATION,
    H1MuJoCoLMIK,
    LMIKResult,
    VRRelativePoseTracker,
    _damped_least_squares_step,
    _load_mujoco,
    _rotation_error,
    _stack_dual_arm_system,
)


def _rotz(angle: float) -> np.ndarray:
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


class VRRelativePoseTrackerTest(unittest.TestCase):
    def setUp(self):
        self.ee0 = np.eye(4)
        self.ee0[:3, 3] = [0.4, 0.2, 0.8]
        self.basis = np.array(
            [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        )

    def test_h1_defaults_match_existing_transform_and_joint_order(self):
        np.testing.assert_allclose(H1_XR_TO_ROBOT_ROTATION, self.basis)
        self.assertEqual(len(H1_LEFT_ARM_JOINT_NAMES), 7)
        self.assertEqual(len(H1_RIGHT_ARM_JOINT_NAMES), 7)
        self.assertTrue(H1_LEFT_ARM_JOINT_NAMES[0].startswith("Robot_Left"))
        self.assertTrue(H1_RIGHT_ARM_JOINT_NAMES[0].startswith("Robot_Right"))

    def test_first_frame_captures_reference_without_motion(self):
        tracker = VRRelativePoseTracker(self.ee0, self.basis)

        target = tracker.update(np.eye(4))

        np.testing.assert_allclose(target, self.ee0)
        self.assertTrue(tracker.initialized)

    def test_translation_uses_h1_basis_and_scale(self):
        tracker = VRRelativePoseTracker(
            self.ee0, self.basis, position_scale=2.0, ema_alpha=0.5
        )
        tracker.update(np.eye(4))
        wrist = np.eye(4)
        wrist[:3, 3] = [0.1, 0.2, 0.3]

        target = tracker.update(wrist)

        expected = self.ee0[:3, 3] + 2.0 * (self.basis @ wrist[:3, 3])
        np.testing.assert_allclose(target[:3, 3], expected)

    def test_translation_ema_blends_later_residuals(self):
        tracker = VRRelativePoseTracker(
            self.ee0, np.eye(3), position_scale=1.0, ema_alpha=0.25
        )
        tracker.update(np.eye(4))
        wrist = np.eye(4)
        wrist[0, 3] = 1.0
        tracker.update(wrist)
        wrist[0, 3] = 3.0

        target = tracker.update(wrist)

        self.assertAlmostEqual(target[0, 3], self.ee0[0, 3] + 1.5)

    def test_rotation_is_relative_to_reference_and_composed_with_ee(self):
        ee0 = self.ee0.copy()
        ee0[:3, :3] = _rotz(-np.pi / 4)
        tracker = VRRelativePoseTracker(ee0, np.eye(3))
        reference = np.eye(4)
        reference[:3, :3] = _rotz(np.pi / 6)
        tracker.update(reference)
        wrist = reference.copy()
        wrist[:3, :3] = _rotz(np.pi / 2) @ reference[:3, :3]

        target = tracker.update(wrist)

        np.testing.assert_allclose(
            target[:3, :3], _rotz(np.pi / 2) @ ee0[:3, :3], atol=1e-7
        )

    def test_position_deadband_zeros_small_residual(self):
        tracker = VRRelativePoseTracker(
            self.ee0, np.eye(3), position_deadband=0.02
        )
        tracker.update(np.eye(4))
        wrist = np.eye(4)
        wrist[0, 3] = 0.01

        np.testing.assert_allclose(tracker.update(wrist), self.ee0)

    def test_rotation_deadband_zeros_small_residual(self):
        tracker = VRRelativePoseTracker(
            self.ee0, np.eye(3), rotation_deadband_deg=1.0
        )
        tracker.update(np.eye(4))
        wrist = np.eye(4)
        wrist[:3, :3] = _rotz(np.deg2rad(0.5))

        np.testing.assert_allclose(tracker.update(wrist), self.ee0, atol=1e-7)

    def test_reset_captures_new_references(self):
        tracker = VRRelativePoseTracker(self.ee0, np.eye(3))
        tracker.update(np.eye(4))
        moved = np.eye(4)
        moved[0, 3] = 1.0
        tracker.update(moved)
        new_ee = self.ee0.copy()
        new_ee[1, 3] += 0.2

        tracker.reset(new_ee)
        target = tracker.update(moved)

        np.testing.assert_allclose(target, new_ee)
        self.assertIsNone(tracker.translation_residual)

    def test_state_accessors_return_defensive_copies(self):
        tracker = VRRelativePoseTracker(self.ee0, np.eye(3))
        result = tracker.update(np.eye(4))
        result[0, 0] = 99.0
        target = tracker.target_pose
        target[0, 0] = 88.0

        self.assertEqual(tracker.target_pose[0, 0], 1.0)

    def test_invalid_configuration_is_rejected(self):
        invalid_kwargs = (
            {"position_scale": 0.0},
            {"ema_alpha": 0.0},
            {"ema_alpha": 1.1},
            {"position_deadband": -0.1},
            {"rotation_deadband_deg": -1.0},
        )
        for kwargs in invalid_kwargs:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                VRRelativePoseTracker(self.ee0, np.eye(3), **kwargs)

    def test_invalid_pose_is_rejected(self):
        tracker = VRRelativePoseTracker(self.ee0, np.eye(3))
        with self.assertRaises(ValueError):
            tracker.update(np.eye(3))
        invalid = np.eye(4)
        invalid[0, 0] = np.nan
        with self.assertRaises(ValueError):
            tracker.update(invalid)


class LMMathTest(unittest.TestCase):
    def test_damped_step_matches_closed_form(self):
        jacobian = np.array([[2.0]])
        error = np.array([1.0])

        actual = _damped_least_squares_step(jacobian, error, damping=0.5)

        np.testing.assert_allclose(actual, [2.0 / 4.5])

    def test_rotation_error_identity_is_zero(self):
        np.testing.assert_allclose(
            _rotation_error(np.eye(3), np.eye(3)), np.zeros(3)
        )

    def test_rotation_error_has_expected_axis_and_angle(self):
        error = _rotation_error(_rotz(np.pi / 2), np.eye(3))
        np.testing.assert_allclose(error, [0.0, 0.0, np.pi / 2], atol=1e-7)

    def test_rotation_error_is_finite_at_pi(self):
        error = _rotation_error(_rotz(np.pi), np.eye(3))
        self.assertTrue(np.all(np.isfinite(error)))
        self.assertAlmostEqual(np.linalg.norm(error), np.pi, places=6)

    def test_stacked_system_orders_left_then_right(self):
        left_error = np.arange(6.0)
        right_error = np.arange(10.0, 16.0)
        left_jacobian = np.full((6, 14), 1.0)
        right_jacobian = np.full((6, 14), 2.0)

        error, jacobian = _stack_dual_arm_system(
            left_error,
            right_error,
            left_jacobian,
            right_jacobian,
            rotation_weight=0.5,
        )

        np.testing.assert_allclose(
            error, [0, 1, 2, 1.5, 2, 2.5, 10, 11, 12, 6.5, 7, 7.5]
        )
        np.testing.assert_allclose(jacobian[:3], 1.0)
        np.testing.assert_allclose(jacobian[3:6], 0.5)
        np.testing.assert_allclose(jacobian[6:9], 2.0)
        np.testing.assert_allclose(jacobian[9:12], 1.0)

    def test_lm_helpers_reject_invalid_parameters(self):
        with self.assertRaises(ValueError):
            _damped_least_squares_step(np.eye(2), np.ones(2), damping=0.0)
        with self.assertRaises(ValueError):
            _stack_dual_arm_system(
                np.ones(5), np.ones(6), np.ones((6, 14)), np.ones((6, 14)), 1.0
            )

    def test_result_fields_are_explicit(self):
        result = LMIKResult(np.zeros(14), True, 3, 1e-5, 2e-5)
        self.assertEqual(result.arm_q.shape, (14,))
        self.assertTrue(result.converged)
        self.assertEqual(result.iterations, 3)


class H1MuJoCoLMIKTest(unittest.TestCase):
    def test_missing_mujoco_has_actionable_error(self):
        with mock.patch.object(
            importlib, "import_module", side_effect=ModuleNotFoundError("mujoco")
        ):
            with self.assertRaisesRegex(RuntimeError, "optional 'mujoco' package"):
                _load_mujoco()

    def test_current_arm_q_must_have_fourteen_values(self):
        solver = object.__new__(H1MuJoCoLMIK)
        with self.assertRaisesRegex(ValueError, "14"):
            solver.solve(np.eye(4), np.eye(4), np.zeros(13))

    def test_solver_parameters_are_validated_before_model_access(self):
        solver = object.__new__(H1MuJoCoLMIK)
        invalid_options = (
            {"max_iters": 0},
            {"tolerance": 0.0},
            {"rotation_weight": -1.0},
            {"damping": 0.0},
        )
        for options in invalid_options:
            with self.subTest(options=options), self.assertRaises(ValueError):
                solver.solve(np.eye(4), np.eye(4), np.zeros(14), **options)


try:
    import mujoco  # type: ignore
except ImportError:
    mujoco = None


@unittest.skipUnless(mujoco is not None, "MuJoCo not installed")
class H1MuJoCoIntegrationTest(unittest.TestCase):
    def test_h1_model_resolves_and_small_dual_arm_target_is_finite(self):
        repo_root = Path(__file__).resolve().parents[1]
        solver = H1MuJoCoLMIK.from_h1_assets(repo_root)
        current_q = np.zeros(14)
        left_target, right_target = solver.forward_kinematics(current_q)
        left_target[0, 3] += 0.005
        right_target[0, 3] += 0.005

        result = solver.solve(
            left_target,
            right_target,
            current_q,
            max_iters=50,
            tolerance=5e-3,
        )

        self.assertEqual(result.arm_q.shape, (14,))
        self.assertTrue(np.all(np.isfinite(result.arm_q)))


if __name__ == "__main__":
    unittest.main()
