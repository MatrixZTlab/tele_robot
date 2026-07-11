# H1 VR Relative Pose and MuJoCo LM IK Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one isolated TOPSTAR_H1 module that reproduces `vr_teleop` fixed-reference wrist tracking and dual-arm MuJoCo damped LM IK without changing the existing live teleoperation path.

**Architecture:** `VRRelativePoseTracker` converts raw TeleVuer OpenXR wrist matrices into fixed-reference H1 end-effector targets using translation EMA and pose deadbands. `H1MuJoCoLMIK` optionally loads the existing H1 URDF, resolves only the 14 arm degrees of freedom, and solves stacked left/right tool pose errors with damped LM. Pure NumPy helpers remain testable when MuJoCo is not installed.

**Tech Stack:** Python 3.10+, NumPy, optional MuJoCo Python package, `unittest`.

## Global Constraints

- Create only one production file: `teleop/robot_control/topstar_h1/vr_mujoco_relative_teleop.py`.
- Do not modify `teleop_hand_and_arm.py`, existing robot drivers, `XRTransformer`, or Pinocchio/CasADi IK.
- Preserve H1 arm order: seven left joints followed by seven right joints.
- Import MuJoCo lazily so importing the new module does not require MuJoCo.
- Do not add workspace clipping, collision checking, or command publication.

---

### Task 1: Fixed-reference VR pose tracker

**Files:**
- Create: `teleop/robot_control/topstar_h1/vr_mujoco_relative_teleop.py`
- Create: `tests/test_vr_mujoco_relative_teleop.py`

**Interfaces:**
- Consumes: OpenXR wrist pose `(4, 4)`, initial H1 end-effector pose `(4, 4)`, OpenXR-to-H1 rotation `(3, 3)`.
- Produces: `VRRelativePoseTracker.update(wrist_pose_xr) -> np.ndarray`, `reset(initial_ee_pose=None) -> None`, and defensive-copy state properties.

- [ ] **Step 1: Write failing tracker tests**

Add `VRRelativePoseTrackerTest` using this common setup:

```python
class VRRelativePoseTrackerTest(unittest.TestCase):
    def setUp(self):
        self.ee0 = np.eye(4)
        self.ee0[:3, 3] = [0.4, 0.2, 0.8]
        self.basis = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])

    def test_first_frame_captures_reference_without_motion(self):
        tracker = VRRelativePoseTracker(self.ee0, self.basis)
        target = tracker.update(np.eye(4))
        np.testing.assert_allclose(target, self.ee0)
        self.assertTrue(tracker.initialized)

    def test_translation_uses_basis_scale_and_ema(self):
        tracker = VRRelativePoseTracker(
            self.ee0, self.basis, position_scale=2.0, ema_alpha=0.5
        )
        tracker.update(np.eye(4))
        wrist = np.eye(4)
        wrist[:3, 3] = [0.1, 0.2, 0.3]
        target = tracker.update(wrist)
        expected = self.ee0[:3, 3] + 2.0 * (self.basis @ wrist[:3, 3])
        np.testing.assert_allclose(target[:3, 3], expected)
```

Add the remaining behaviors as explicit tests:

```python
def test_rotation_is_relative_to_reference_and_composed_with_ee(self):
    tracker = VRRelativePoseTracker(self.ee0, np.eye(3))
    tracker.update(np.eye(4))
    wrist = np.eye(4)
    wrist[:3, :3] = _rotz(np.pi / 2)
    np.testing.assert_allclose(tracker.update(wrist)[:3, :3], _rotz(np.pi / 2), atol=1e-7)

def test_position_deadband_zeros_small_residual(self):
    tracker = VRRelativePoseTracker(self.ee0, np.eye(3), position_deadband=0.02)
    tracker.update(np.eye(4))
    wrist = np.eye(4); wrist[0, 3] = 0.01
    np.testing.assert_allclose(tracker.update(wrist), self.ee0)

def test_reset_captures_a_new_wrist_reference(self):
    tracker = VRRelativePoseTracker(self.ee0, np.eye(3))
    tracker.update(np.eye(4))
    new_ee = self.ee0.copy(); new_ee[0, 3] += 0.2
    tracker.reset(new_ee)
    wrist = np.eye(4); wrist[1, 3] = 0.4
    np.testing.assert_allclose(tracker.update(wrist), new_ee)

def test_returned_target_is_a_defensive_copy(self):
    tracker = VRRelativePoseTracker(self.ee0, np.eye(3))
    result = tracker.update(np.eye(4)); result[0, 0] = 99
    self.assertNotEqual(tracker.target_pose[0, 0], 99)

def test_invalid_configuration_and_pose_are_rejected(self):
    with self.assertRaises(ValueError):
        VRRelativePoseTracker(self.ee0, np.eye(3), ema_alpha=0.0)
    tracker = VRRelativePoseTracker(self.ee0, np.eye(3))
    with self.assertRaises(ValueError):
        tracker.update(np.eye(3))
```

Add a rotation-deadband test with a `0.5°` wrist rotation and a `1.0°`
threshold; its target rotation must remain the captured end-effector rotation.

- [ ] **Step 2: Run the tracker tests and verify RED**

Run:

```bash
python -m unittest tests.test_vr_mujoco_relative_teleop.VRRelativePoseTrackerTest -v
```

Expected: import failure because `vr_mujoco_relative_teleop.py` does not exist.

- [ ] **Step 3: Implement the minimal tracker**

Create these helpers and class in the production file:

Implement `_validate_pose`, `_validate_rotation`, and `_rotation_angle` with
shape, finite-value, homogeneous-row, orthonormality, and determinant checks.
Implement `VRRelativePoseTracker` with the exact public signatures from the
design. `update()` must follow this body order:

```python
wrist = _validate_pose(wrist_pose_xr, "wrist_pose_xr")
if self._reference_wrist_pose is None:
    self._reference_wrist_pose = wrist
    return self.target_pose

delta = self._basis @ (wrist[:3, 3] - self._reference_wrist_pose[:3, 3])
if self._smoothed_translation is None:
    self._smoothed_translation = delta
else:
    self._smoothed_translation = (
        self._ema_alpha * delta
        + (1.0 - self._ema_alpha) * self._smoothed_translation
    )
if np.linalg.norm(self._smoothed_translation) < self._position_deadband:
    self._smoothed_translation = np.zeros(3)

delta_r_xr = wrist[:3, :3] @ self._reference_wrist_pose[:3, :3].T
delta_r = self._basis @ delta_r_xr @ self._basis.T
if _rotation_angle(delta_r) < self._rotation_deadband_rad:
    delta_r = np.eye(3)

self._target_pose = self._initial_ee_pose.copy()
self._target_pose[:3, 3] += self._position_scale * self._smoothed_translation
self._target_pose[:3, :3] = delta_r @ self._initial_ee_pose[:3, :3]
return self.target_pose
```

- [ ] **Step 4: Run tracker tests and verify GREEN**

Run the Task 1 test command. Expected: all tracker tests pass.

- [ ] **Step 5: Commit Task 1**

```bash
git add teleop/robot_control/topstar_h1/vr_mujoco_relative_teleop.py tests/test_vr_mujoco_relative_teleop.py
git commit -m "feat: add H1 VR relative pose tracker"
```

### Task 2: Pure NumPy damped LM math

**Files:**
- Modify: `teleop/robot_control/topstar_h1/vr_mujoco_relative_teleop.py`
- Modify: `tests/test_vr_mujoco_relative_teleop.py`

**Interfaces:**
- Consumes: stacked pose error and Jacobian arrays.
- Produces: `_rotation_error(target, current)`, `_stack_dual_arm_system(left_error, right_error, left_jacobian, right_jacobian, rotation_weight)`, `_damped_least_squares_step(jacobian, error, damping)` and immutable `LMIKResult`.

- [ ] **Step 1: Write failing LM math tests**

```python
class LMMathTest(unittest.TestCase):
    def test_damped_step_matches_closed_form(self):
        jac = np.array([[2.0]])
        err = np.array([1.0])
        actual = _damped_least_squares_step(jac, err, damping=0.5)
        np.testing.assert_allclose(actual, [2.0 / 4.5])

    def test_rotation_error_identity_is_zero(self):
        np.testing.assert_allclose(_rotation_error(np.eye(3), np.eye(3)), 0.0)

    def test_stacked_system_orders_left_then_right(self):
        left_error = np.arange(6.0)
        right_error = np.arange(10.0, 16.0)
        left_jac = np.full((6, 14), 1.0)
        right_jac = np.full((6, 14), 2.0)
        error, jac = _stack_dual_arm_system(
            left_error, right_error, left_jac, right_jac, rotation_weight=0.5
        )
        np.testing.assert_allclose(error, [0, 1, 2, 1.5, 2, 2.5, 10, 11, 12, 6.5, 7, 7.5])
        np.testing.assert_allclose(jac[:3], 1.0)
        np.testing.assert_allclose(jac[3:6], 0.5)
        np.testing.assert_allclose(jac[6:9], 2.0)
        np.testing.assert_allclose(jac[9:12], 1.0)
```

Add tests for a known 90-degree z rotation, invalid damping, mismatched shapes,
and `LMIKResult` fields.

- [ ] **Step 2: Run LM math tests and verify RED**

Run:

```bash
python -m unittest tests.test_vr_mujoco_relative_teleop.LMMathTest -v
```

Expected: import failures for the missing LM helpers.

- [ ] **Step 3: Implement minimal LM helpers**

```python
@dataclass(frozen=True)
class LMIKResult:
    arm_q: np.ndarray
    converged: bool
    iterations: int
    translation_error_norm: float
    rotation_error_norm: float

def _damped_least_squares_step(jacobian, error, damping):
    return jacobian.T @ np.linalg.solve(
        jacobian @ jacobian.T + damping * np.eye(jacobian.shape[0]),
        error,
    )
```

Implement robust SO(3) rotation error and explicit stacked-system validation.

- [ ] **Step 4: Run Task 1 and Task 2 tests**

Run:

```bash
python -m unittest tests.test_vr_mujoco_relative_teleop -v
```

Expected: tracker and LM math tests pass.

- [ ] **Step 5: Commit Task 2**

```bash
git add teleop/robot_control/topstar_h1/vr_mujoco_relative_teleop.py tests/test_vr_mujoco_relative_teleop.py
git commit -m "feat: add dual-arm damped LM math"
```

### Task 3: Optional MuJoCo-backed H1 dual-arm solver

**Files:**
- Modify: `teleop/robot_control/topstar_h1/vr_mujoco_relative_teleop.py`
- Modify: `tests/test_vr_mujoco_relative_teleop.py`

**Interfaces:**
- Consumes: H1 URDF, 14 configured arm joint names, wrist-body names and `0.03 m` local tool offset.
- Produces: `H1MuJoCoLMIK.from_urdf(urdf_path, left_joint_names, right_joint_names, left_ee_body, right_ee_body, ee_offset)` and `solve(left_target_pose, right_target_pose, current_arm_q, **solver_options) -> LMIKResult`.

- [ ] **Step 1: Write failing optional-dependency and adapter tests**

```python
def test_missing_mujoco_has_actionable_error(self):
    with mock.patch.dict(sys.modules, {"mujoco": None}):
        with self.assertRaisesRegex(RuntimeError, "optional 'mujoco' package"):
            _load_mujoco()

def test_current_arm_q_must_have_fourteen_values(self):
    solver = _make_fake_solver()
    with self.assertRaisesRegex(ValueError, "14"):
        solver.solve(np.eye(4), np.eye(4), np.zeros(13))

def test_solver_parameter_validation(self):
    solver = _make_fake_solver()
    with self.assertRaises(ValueError):
        solver.solve(np.eye(4), np.eye(4), np.zeros(14), damping=0.0)
```

Add an integration test decorated with `unittest.skipUnless(mujoco_available,
"MuJoCo not installed")`. It loads `H1RobotConfig().urdf_path`, resolves every
configured arm joint, captures both wrist tool poses, requests a small reachable
translation, solves, and asserts finite 14-element output while non-arm qpos are
unchanged in the solver workspace.

- [ ] **Step 2: Run adapter tests and verify RED**

Run:

```bash
python -m unittest tests.test_vr_mujoco_relative_teleop.H1MuJoCoLMIKTest -v
```

Expected: failure because `H1MuJoCoLMIK` is missing.

- [ ] **Step 3: Implement lazy MuJoCo adapter**

Add:

Implement `_load_mujoco()` with an actionable `RuntimeError` on import failure.
Implement `H1MuJoCoLMIK.from_urdf()` with explicit parameters `urdf_path`,
`left_joint_names`, `right_joint_names`, `left_ee_body`, `right_ee_body`, and
`ee_offset=(0.0, 0.0, 0.03)`. Implement `solve()` with keyword parameters
`max_iters=30`, `tolerance=1e-4`, `rotation_weight=1.0`, and `damping=1e-3`.

Resolve joint qpos/dof addresses by name. Use `mj_forward`, transform the local
tool offsets through each wrist body's current rotation, call `mj_jac` at those
world tool points, zero all non-arm columns, stack both arms, compute one damped
LM step, call `mj_integratePos`, and clip ranged arm joints.

- [ ] **Step 4: Run all available tests**

Run:

```bash
python -m unittest tests.test_vr_mujoco_relative_teleop -v
python -m unittest discover -s tests -v
```

Expected: all unit tests pass; the H1 MuJoCo integration test either passes or
is explicitly skipped only because MuJoCo is not installed.

- [ ] **Step 5: Run static checks**

```bash
python -m compileall teleop/robot_control/topstar_h1/vr_mujoco_relative_teleop.py tests/test_vr_mujoco_relative_teleop.py
git diff --check
```

Expected: both commands exit zero.

- [ ] **Step 6: Commit Task 3**

```bash
git add teleop/robot_control/topstar_h1/vr_mujoco_relative_teleop.py tests/test_vr_mujoco_relative_teleop.py
git commit -m "feat: add optional H1 MuJoCo LM IK"
```

### Task 4: Final review and handoff

**Files:**
- Review: `teleop/robot_control/topstar_h1/vr_mujoco_relative_teleop.py`
- Review: `tests/test_vr_mujoco_relative_teleop.py`
- Review: `docs/superpowers/specs/2026-07-11-vr-relative-pose-tracker-design.md`

**Interfaces:**
- Consumes: the completed module and tests.
- Produces: verified review findings and Linux MuJoCo validation commands.

- [ ] **Step 1: Review requirements against implementation**

Check fixed-reference semantics, H1 coordinate mapping, 14-joint order, lazy
MuJoCo import, no existing-path modifications, no hidden solver state, and
defensive copies.

- [ ] **Step 2: Review numerical behavior**

Check SO(3) behavior near zero and pi, damping sign, Jacobian/error ordering,
tool-point Jacobians, joint-address mapping, convergence reporting, and failure
messages.

- [ ] **Step 3: Re-run fresh verification**

Run the complete test discovery, compileall, and `git diff --check` commands
immediately before reporting completion.

- [ ] **Step 4: Document Linux validation command**

```bash
python -m pip install mujoco
python -m unittest tests.test_vr_mujoco_relative_teleop.H1MuJoCoIntegrationTest -v
```

Expected: the H1 URDF loads, 14 arm joints resolve, and the small reachable
dual-arm test converges.
