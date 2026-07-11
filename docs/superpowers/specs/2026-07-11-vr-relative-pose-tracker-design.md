# H1 VR Relative Pose and MuJoCo LM IK Design

## Goal

Add the `vr_teleop` wrist-relative pose method and damped Levenberg-Marquardt
MuJoCo IK method to `tele_robot` as an isolated experimental module for
`TOPSTAR_H1`. The existing teleoperation loop, `XRTransformer`, robot driver,
Pinocchio/CasADi IK solver, and command path remain unchanged.

## Scope

Production code is added only in
`teleop/robot_control/topstar_h1/vr_mujoco_relative_teleop.py`. Unit tests live
in `tests/test_vr_mujoco_relative_teleop.py`.

The module consumes:

- a raw OpenXR wrist pose from `TeleVuerWrapper`, represented as a finite
  homogeneous `4 x 4` matrix;
- the H1 end-effector pose captured by FK when tracking starts, represented as
  a finite homogeneous `4 x 4` matrix;
- the H1 OpenXR-to-robot basis rotation.

It produces homogeneous `4 x 4` left and right target end-effector poses in the
robot frame. The same module can solve those targets with a MuJoCo-based,
dual-arm damped LM solver and return the 14 H1 arm joint targets.

This change does not connect the module to the live robot. It does not add
workspace clipping, collision checking, or velocity limiting. MuJoCo remains
an optional runtime dependency and is imported only when the IK class is
constructed, so the existing teleoperation package remains importable without
MuJoCo.

## Interface

`VRRelativePoseTracker` has the following public API:

```python
tracker = VRRelativePoseTracker(
    initial_ee_pose,
    xr_to_robot_rotation,
    position_scale=1.5,
    ema_alpha=0.8,
    position_deadband=0.0,
    rotation_deadband_deg=0.0,
)

target = tracker.update(current_wrist_pose_xr)
tracker.reset(new_initial_ee_pose)
```

The first valid `update()` after construction or `reset()` captures the wrist
reference and returns the initial end-effector pose unchanged. Later calls
return the updated target pose. Read-only properties expose `initialized`,
`target_pose`, `translation_residual`, and `rotation_residual`.

Invalid shapes, non-finite matrices, invalid homogeneous last rows, invalid
rotation shapes, `position_scale <= 0`, `ema_alpha` outside `(0, 1]`, and
negative deadbands raise `ValueError`.

`H1MuJoCoLMIK` has the following public API:

```python
ik = H1MuJoCoLMIK.from_urdf(
    urdf_path,
    left_joint_names,
    right_joint_names,
    left_ee_body="Robot_Left_Hand_6_Link",
    right_ee_body="Robot_Right_Hand_6_Link",
    ee_offset=(0.0, 0.0, 0.03),
)

result = ik.solve(
    left_target_pose,
    right_target_pose,
    current_arm_q,
    max_iters=30,
    tolerance=1e-4,
    rotation_weight=1.0,
    damping=1e-3,
)
```

`current_arm_q` and `result.arm_q` use the existing H1 order: seven left-arm
joints followed by seven right-arm joints. `result` also reports convergence,
iterations, translation error norm, and rotation error norm. Missing MuJoCo,
missing model bodies or joints, invalid joint dimensions, and invalid solver
parameters raise explicit exceptions.

## Pose Calculation

Let the captured wrist pose be `(R_w0, p_w0)`, the current wrist pose be
`(R_w, p_w)`, the captured H1 end-effector pose be `(R_e0, p_e0)`, and the H1
basis rotation be `B`.

Translation follows the `vr_teleop` residual tracker:

```text
delta_p_xr = p_w - p_w0
delta_p_robot = B * delta_p_xr
smoothed_t = delta_p_robot                              # first residual
smoothed_t = alpha * delta_p_robot + (1-alpha) * old   # later residuals
p_target = p_e0 + position_scale * smoothed_t
```

If the norm of `smoothed_t` is below the translation deadband, it becomes zero.

Rotation is relative to the captured wrist orientation:

```text
delta_R_xr = R_w * transpose(R_w0)
delta_R_robot = B * delta_R_xr * transpose(B)
R_target = delta_R_robot * R_e0
```

If the angle of `delta_R_robot` is below the rotation deadband, the residual is
replaced with identity. Rotation is not EMA-smoothed, matching `vr_teleop`.

This is a fixed-reference method. It never integrates adjacent-frame motion.

## MuJoCo LM IK

The H1 solver follows `vr_teleop` but solves both arms in one system. Each
iteration performs MuJoCo forward kinematics, computes left and right
end-effector pose errors, and computes translational and rotational Jacobians
at each tool point. The fixed tool point is `0.03 m` along the local z axis of
`Robot_Left_Hand_6_Link` or `Robot_Right_Hand_6_Link`, matching the H1 URDF
fixed end-effector joints.

Only the 14 named H1 arm degrees of freedom are enabled. All wheel, torso, head,
and unrelated columns are zeroed. The stacked update is:

```text
error = [left_position_error,
         rotation_weight * left_rotation_error,
         right_position_error,
         rotation_weight * right_rotation_error]

J = stack(left_position_J,
          rotation_weight * left_rotation_J,
          right_position_J,
          rotation_weight * right_rotation_J)

dq = transpose(J) * inverse(J * transpose(J) + damping * I) * error
```

MuJoCo integrates `dq` into its full generalized position vector. The solver
then clips the 14 arm positions to MuJoCo's joint ranges when ranges are
available. The next control frame uses the measured/current arm state as the
initial guess; the solver does not retain hidden integrated state between calls.

## Testing

The test suite uses `unittest` and NumPy. Tracker tests verify:

- first-frame reference capture without target motion;
- scaled translation in the H1 robot basis;
- translation EMA behavior;
- relative rotation composition;
- translation and rotation deadbands;
- reset and recapture behavior;
- defensive copies of returned state;
- invalid configuration and invalid pose rejection.

The LM math tests verify:

- damped least-squares updates for a known Jacobian;
- stacked left/right error ordering;
- rotation error for identity and a known axis-angle rotation;
- parameter and shape validation;
- explicit optional-dependency failure when MuJoCo is unavailable.

An integration test that loads the H1 URDF and checks dual-arm convergence is
skipped when MuJoCo is unavailable. In a Linux environment with MuJoCo
installed, it must resolve the 14 named arm joints, solve a small reachable
dual-arm displacement, preserve non-arm joints, and report convergence.

No ROS2, CasADi, Pinocchio, robot SDK, or XR hardware is required for the unit
tests.
