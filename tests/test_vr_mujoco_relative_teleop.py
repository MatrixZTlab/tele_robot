import importlib
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from teleop.robot_control.vr_mujoco_relative_teleop import (
    H1_HARD_ARM_LOWER,
    H1_HARD_ARM_UPPER,
    H1_SAFE_ARM_LOWER,
    H1_SAFE_ARM_UPPER,
    H1_LEFT_ARM_JOINT_NAMES,
    H1_RIGHT_ARM_JOINT_NAMES,
    H1_XR_TO_ROBOT_ROTATION,
    H1MuJoCoLMIK,
    LMIKResult,
    VRRelativePoseTracker,
    _augment_regularization,
    _damped_least_squares_step,
    _load_mujoco,
    _rotation_error,
    _stack_dual_arm_system,
)
from teleop.robot_control.topstar_h1.joint_convention import (
    H1_ARM_HW_TO_MODEL_SIGN,
    H1_MUJOCO_URDF_FILENAME,
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

    def test_translation_axis_sign_flips_only_robot_lateral_axis(self):
        tracker = VRRelativePoseTracker(
            self.ee0,
            self.basis,
            position_scale=1.0,
            ema_alpha=1.0,
            translation_axis_sign=(1.0, -1.0, 1.0),
        )
        tracker.update(np.eye(4))
        wrist = np.eye(4)
        wrist[:3, 3] = [0.1, 0.2, 0.3]

        target = tracker.update(wrist)

        ordinary = self.basis @ wrist[:3, 3]
        expected = ordinary * np.array([1.0, -1.0, 1.0])
        np.testing.assert_allclose(target[:3, 3], self.ee0[:3, 3] + expected)
        self.assertAlmostEqual(target[0, 3] - self.ee0[0, 3], ordinary[0])
        self.assertAlmostEqual(target[2, 3] - self.ee0[2, 3], ordinary[2])

    def test_translation_axis_sign_rejects_non_mirror_values(self):
        with self.assertRaisesRegex(ValueError, "translation_axis_sign"):
            VRRelativePoseTracker(
                self.ee0, self.basis, translation_axis_sign=(1.0, 0.0, 1.0)
            )

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
        invalid_last_row = np.eye(4)
        invalid_last_row[3, 0] = 1.0
        with self.assertRaises(ValueError):
            tracker.update(invalid_last_row)

    def test_negate_rot_xy_flips_x_and_y_rotation_components(self):
        # vr_teleop WristTracker's negate_rot_xy negates (qx, qy) of the
        # relative rotation, i.e. R -> diag(-1,-1,1) @ R @ diag(-1,-1,1).
        # A rotation about X should have its sense flipped; a rotation about Z
        # should be unchanged.
        neg = np.diag([-1.0, -1.0, 1.0])

        def rotx(angle):
            c, s = np.cos(angle), np.sin(angle)
            return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])

        plain = VRRelativePoseTracker(self.ee0, np.eye(3))
        flipped = VRRelativePoseTracker(self.ee0, np.eye(3), negate_rot_xy=True)
        plain.update(np.eye(4))
        flipped.update(np.eye(4))
        wrist = np.eye(4)
        wrist[:3, :3] = rotx(np.pi / 5)

        plain.update(wrist)
        flipped.update(wrist)

        np.testing.assert_allclose(
            flipped.rotation_residual,
            neg @ plain.rotation_residual @ neg,
            atol=1e-12,
        )
        # X-axis rotation sense is flipped by the negation.
        np.testing.assert_allclose(
            flipped.rotation_residual, rotx(-np.pi / 5), atol=1e-7
        )

    def test_negate_rot_xy_defaults_to_false(self):
        # Default must match vr_teleop's negate_rot_xy=False: no sign flip.
        plain = VRRelativePoseTracker(self.ee0, np.eye(3), negate_rot_xy=False)
        default = VRRelativePoseTracker(self.ee0, np.eye(3))
        for tracker in (plain, default):
            tracker.update(np.eye(4))
        wrist = np.eye(4)
        wrist[:3, :3] = _rotz(np.pi / 3)
        np.testing.assert_allclose(
            plain.update(wrist), default.update(wrist), atol=1e-12
        )

    def test_non_orthonormal_wrist_is_normalized_to_valid_target_rotation(self):
        # The public interface accepts small XR matrix drift, but the internal
        # quaternion path must normalize it before returning a 4x4 IK target.
        tracker = VRRelativePoseTracker(self.ee0, np.eye(3))
        tracker.update(np.eye(4))
        drifted = np.eye(4)
        drifted[:3, :3] = _rotz(0.2)
        drifted[0, 0] += 1e-3

        target = tracker.update(drifted)

        rotation = target[:3, :3]
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-10)
        self.assertAlmostEqual(np.linalg.det(rotation), 1.0, places=10)

    def test_quaternion_internal_path_preserves_known_relative_rotation(self):
        tracker = VRRelativePoseTracker(self.ee0, self.basis)
        reference = np.eye(4)
        reference[:3, :3] = _rotz(-0.3)
        tracker.update(reference)
        current = reference.copy()
        current[:3, :3] = _rotz(0.4) @ reference[:3, :3]

        target = tracker.update(current)

        expected_delta = self.basis @ _rotz(0.4) @ self.basis.T
        np.testing.assert_allclose(
            target[:3, :3], expected_delta @ self.ee0[:3, :3], atol=1e-10
        )


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

    def test_rotation_error_near_pi_has_correct_axis_direction(self):
        # Near-pi branch: the recovered axis must point the physically correct
        # way (disambiguated by the antisymmetric part), not an arbitrary
        # eigenvector sign. A +Z rotation just under pi must yield a +Z axis.
        angle = np.pi - 5e-7
        error = _rotation_error(_rotz(angle), np.eye(3))
        np.testing.assert_allclose(error, [0.0, 0.0, angle], atol=1e-6)
        # And a -Z rotation just under pi must yield a -Z axis.
        error_neg = _rotation_error(_rotz(-angle), np.eye(3))
        np.testing.assert_allclose(error_neg, [0.0, 0.0, -angle], atol=1e-6)

    def test_rotation_error_accepts_slightly_non_orthonormal_input(self):
        # Inner-loop _rotation_error no longer re-validates orthonormality;
        # MuJoCo FK output with tiny numerical drift must not raise.
        drifted = _rotz(0.3)
        drifted[0, 0] += 1e-6
        error = _rotation_error(drifted, np.eye(3))
        self.assertTrue(np.all(np.isfinite(error)))

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

    def test_regularization_is_noop_with_original_defaults(self):
        # vr_teleop defaults: home_qpos=None, current_q_weight=0.0 -> unchanged.
        task_error = np.arange(12.0)
        task_jac = np.ones((12, 14))
        error, jacobian = _augment_regularization(
            task_error,
            task_jac,
            current_q=np.zeros(14),
            initial_q=np.zeros(14),
            home_qpos=None,
            home_weight=0.01,
            current_q_weight=0.0,
        )
        np.testing.assert_array_equal(error, task_error)
        np.testing.assert_array_equal(jacobian, task_jac)

    def test_home_regularization_appends_scaled_pullback_rows(self):
        # Matches solve_pose_ik: appends sqrt(home_weight)*(home - q) residual
        # rows with an identity Jacobian block against the 14 arm DOFs.
        task_error = np.zeros(12)
        task_jac = np.zeros((12, 14))
        current_q = np.full(14, 0.5)
        home_qpos = np.zeros(14)
        error, jacobian = _augment_regularization(
            task_error,
            task_jac,
            current_q=current_q,
            initial_q=current_q,
            home_qpos=home_qpos,
            home_weight=0.04,
            current_q_weight=0.0,
        )
        scale = np.sqrt(0.04)
        self.assertEqual(error.shape, (26,))
        self.assertEqual(jacobian.shape, (26, 14))
        np.testing.assert_allclose(error[12:], scale * (home_qpos - current_q))
        np.testing.assert_allclose(jacobian[12:], scale * np.eye(14))

    def test_current_q_regularization_penalizes_deviation_from_initial(self):
        # current term penalizes deviation from the solve's INITIAL arm state,
        # not the per-iteration q, matching solve_pose_ik's q_init usage.
        task_error = np.zeros(12)
        task_jac = np.zeros((12, 14))
        current_q = np.full(14, 0.3)
        initial_q = np.zeros(14)
        error, jacobian = _augment_regularization(
            task_error,
            task_jac,
            current_q=current_q,
            initial_q=initial_q,
            home_qpos=None,
            home_weight=0.01,
            current_q_weight=0.09,
        )
        scale = np.sqrt(0.09)
        self.assertEqual(error.shape, (26,))
        np.testing.assert_allclose(error[12:], scale * (initial_q - current_q))
        np.testing.assert_allclose(jacobian[12:], scale * np.eye(14))


class H1MuJoCoLMIKTest(unittest.TestCase):
    def test_h1_asset_limit_mode_selects_requested_envelope(self):
        sentinel = object()
        cases = (
            ("safe", H1_SAFE_ARM_LOWER, H1_SAFE_ARM_UPPER),
            ("hard", H1_HARD_ARM_LOWER, H1_HARD_ARM_UPPER),
        )
        for mode, expected_lower, expected_upper in cases:
            with self.subTest(mode=mode), mock.patch.object(
                H1MuJoCoLMIK, "from_urdf", return_value=sentinel
            ) as loader:
                result = H1MuJoCoLMIK.from_h1_assets(
                    Path("/tmp/h1"), arm_limit_mode=mode
                )
                self.assertIs(result, sentinel)
                self.assertEqual(
                    loader.call_args.args[0].name,
                    H1_MUJOCO_URDF_FILENAME,
                )
                np.testing.assert_allclose(
                    loader.call_args.kwargs["arm_joint_lower"], expected_lower
                )
                np.testing.assert_allclose(
                    loader.call_args.kwargs["arm_joint_upper"], expected_upper
                )
        with self.assertRaisesRegex(ValueError, "arm_limit_mode"):
            H1MuJoCoLMIK.from_h1_assets(
                Path("/tmp/h1"), arm_limit_mode="disabled"
            )

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
            {"home_weight": -1.0},
            {"current_q_weight": -1.0},
            {"home_qpos": np.zeros(13)},
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
    def test_revised_fk_matches_legacy_fk_after_left_sign_conversion(self):
        repo_root = Path(__file__).resolve().parents[1]
        model_dir = repo_root / "assets" / "topstar_h1"
        revised = H1MuJoCoLMIK.from_h1_assets(
            repo_root, arm_limit_mode="hard"
        )
        legacy = H1MuJoCoLMIK.from_urdf(
            model_dir / "_tmp_h1_mujoco.urdf",
            H1_LEFT_ARM_JOINT_NAMES,
            H1_RIGHT_ARM_JOINT_NAMES,
            left_ee_body="Robot_Left_Hand_6_Link",
            right_ee_body="Robot_Right_Hand_6_Link",
            ee_offset=(0.0, 0.0, 0.03),
        )
        revised_q = np.array(
            [
                -1.0, -0.5, 0.4, -0.8, 0.7, -0.6, 0.5,
                1.0, -0.5, -0.4, -0.8, -0.7, -0.6, -0.5,
            ]
        )
        legacy_sign = H1_ARM_HW_TO_MODEL_SIGN.copy()
        legacy_sign[[3, 5]] = -1.0
        legacy_q = revised_q * legacy_sign

        revised_left, revised_right = revised.forward_kinematics(revised_q)
        legacy_left, legacy_right = legacy.forward_kinematics(legacy_q)

        np.testing.assert_allclose(revised_left, legacy_left, atol=1e-8)
        np.testing.assert_allclose(revised_right, legacy_right, atol=1e-8)

    def test_h1_assets_use_proven_hardware_limits_not_raw_urdf_ranges(self):
        repo_root = Path(__file__).resolve().parents[1]
        solver = H1MuJoCoLMIK.from_h1_assets(repo_root)

        lower, upper = solver.arm_joint_limits
        np.testing.assert_allclose(lower, H1_SAFE_ARM_LOWER)
        np.testing.assert_allclose(upper, H1_SAFE_ARM_UPPER)
        # This measured left base angle occurs in valid H1 recordings but the
        # repository URDF incorrectly clips it at -1.57 rad.
        recorded_q = np.zeros(14)
        recorded_q[0] = -2.0
        self.assertTrue(solver.arm_q_within_limits(recorded_q))

        lower[0] = 99.0
        self.assertNotEqual(solver.arm_joint_limits[0][0], 99.0)

    def test_h1_joint_clipping_and_limit_tolerance(self):
        repo_root = Path(__file__).resolve().parents[1]
        solver = H1MuJoCoLMIK.from_h1_assets(repo_root)
        lower, upper = solver.arm_joint_limits
        arm_q = np.zeros(14)
        arm_q[0] = -2.0
        arm_q[5] = 10.0
        qpos = solver._full_qpos(arm_q)

        solver._clip_arm_joints(qpos)

        clipped = qpos[solver._qpos_addresses]
        self.assertEqual(clipped[0], -2.0)
        self.assertEqual(clipped[5], upper[5])
        just_outside = np.zeros(14)
        just_outside[5] = upper[5] + 0.01
        self.assertFalse(solver.arm_q_within_limits(just_outside))
        self.assertTrue(
            solver.arm_q_within_limits(just_outside, tolerance=0.02)
        )

    def test_gravity_compensation_is_finite_and_nonzero(self):
        repo_root = Path(__file__).resolve().parents[1]
        solver = H1MuJoCoLMIK.from_h1_assets(repo_root)

        tau = solver.gravity_compensation(np.zeros(14))

        self.assertEqual(tau.shape, (14,))
        self.assertTrue(np.all(np.isfinite(tau)))
        self.assertGreater(float(np.max(np.abs(tau))), 1.0)

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
