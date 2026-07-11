# VR Relative Pose Tracker Design

## Goal

Add the `vr_teleop` wrist-relative pose method to `tele_robot` as an isolated,
matrix-native module for `TOPSTAR_H1`. The existing teleoperation loop,
`XRTransformer`, robot driver, IK solver, and command path remain unchanged.

## Scope

Production code is added only in
`teleop/robot_control/_base/vr_relative_pose_tracker.py`. Unit tests live in
`tests/test_vr_relative_pose_tracker.py`.

The module consumes:

- a raw OpenXR wrist pose from `TeleVuerWrapper`, represented as a finite
  homogeneous `4 x 4` matrix;
- the H1 end-effector pose captured by FK when tracking starts, represented as
  a finite homogeneous `4 x 4` matrix;
- the H1 OpenXR-to-robot basis rotation.

It produces a homogeneous `4 x 4` target end-effector pose in the robot frame,
ready to be passed to the existing H1 IK solver by a future integration change.

This change does not connect the module to the live robot. It does not add
workspace clipping, collision checking, velocity limiting, or a MuJoCo IK
backend.

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

## Pose Calculation

Let the captured wrist pose be `(R_w0, p_w0)`, the current wrist pose be
`(R_w, p_w)`, the captured H1 end-effector pose be `(R_e0, p_e0)`, and the H1
basis rotation be `B`.

Translation follows the `vr_teleop` residual tracker:

```text
delta_p_xr = p_w - p_w0
delta_p_robot = B * delta_p_xr
smoothed_t = alpha * delta_p_robot + (1 - alpha) * previous_smoothed_t
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

## Testing

The test suite uses `unittest` and NumPy only. It verifies:

- first-frame reference capture without target motion;
- scaled translation in the H1 robot basis;
- translation EMA behavior;
- relative rotation composition;
- translation and rotation deadbands;
- reset and recapture behavior;
- defensive copies of returned state;
- invalid configuration and invalid pose rejection.

No ROS2, CasADi, Pinocchio, robot SDK, or XR hardware is required for these
tests.

