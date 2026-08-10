"""Shared joint convention for the revised TOPSTAR H1 model.

The vendor's ``Topstar_Revise.urdf`` and ``joint_defs.py`` define every arm
joint in the same sign convention as the hardware.  Teleoperation arrays in
this repository are ordered as robot-left J1..J7 followed by robot-right
J1..J7, so the vendor's right-first motor-slot order is rearranged here.
"""

from __future__ import annotations

import math

import numpy as np


H1_MODEL_REVISION = "Topstar_Revise_2026-07-15"
H1_ARM_COORDINATE_CONVENTION = "hardware_aligned_left_then_right"
H1_LEGACY_ARM_COORDINATE_CONVENTION = "legacy_left_j4_j6_inverted"
H1_MUJOCO_URDF_FILENAME = "Topstar_Revise_mujoco.urdf"

# The revised URDF uses the hardware sign for all fourteen arm joints.
H1_ARM_HW_TO_MODEL_SIGN = np.ones(14, dtype=np.float64)

# Trajectories recorded with the previous URDF stored robot-left J4 and J6 in
# the old model convention.  Multiplying by this vector upgrades those samples
# to the revised, hardware-aligned convention.  Robot-right values are
# unchanged.
H1_LEGACY_ARM_TO_CURRENT_SIGN = np.ones(14, dtype=np.float64)
H1_LEGACY_ARM_TO_CURRENT_SIGN[[3, 5]] = -1.0


def arm_positions_to_current(q, source_convention: str):
    """Return a 14-DOF H1 arm vector in the revised coordinate convention."""
    values = np.asarray(q, dtype=np.float64)
    if values.shape != (14,):
        raise ValueError(f"expected 14 arm joints, got shape {values.shape}")
    if source_convention == H1_ARM_COORDINATE_CONVENTION:
        return values.copy()
    if source_convention == H1_LEGACY_ARM_COORDINATE_CONVENTION:
        return values * H1_LEGACY_ARM_TO_CURRENT_SIGN
    raise ValueError(f"unsupported H1 arm coordinate convention: {source_convention!r}")

_ONE_ARM_HARD_LOWER = np.array(
    [
        -2.61799388,  # shoulder base, -150 deg
        -1.57079633,  # shoulder, -90 deg
        -2.61799388,  # elbow yaw, -150 deg
        -1.79768913,  # elbow, -103 deg
        -2.87979327,  # wrist yaw, -165 deg
        -1.53588974,  # wrist pitch, -88 deg
        -2.96705973,  # wrist roll, -170 deg
    ],
    dtype=np.float64,
)
_ONE_ARM_HARD_UPPER = np.array(
    [
        2.61799388,   # shoulder base, 150 deg
        0.43633231,   # shoulder, 25 deg
        2.61799388,   # elbow yaw, 150 deg
        0.43633231,   # elbow, 25 deg
        2.87979327,   # wrist yaw, 165 deg
        0.43633231,   # wrist pitch, 25 deg
        2.96705973,   # wrist roll, 170 deg
    ],
    dtype=np.float64,
)

H1_ARM_HARD_LOWER = np.concatenate(
    [_ONE_ARM_HARD_LOWER, _ONE_ARM_HARD_LOWER]
)
H1_ARM_HARD_UPPER = np.concatenate(
    [_ONE_ARM_HARD_UPPER, _ONE_ARM_HARD_UPPER]
)

H1_ARM_SAFETY_MARGIN_RAD = math.radians(5.0)
H1_ARM_SAFE_LOWER = H1_ARM_HARD_LOWER + H1_ARM_SAFETY_MARGIN_RAD
H1_ARM_SAFE_UPPER = H1_ARM_HARD_UPPER - H1_ARM_SAFETY_MARGIN_RAD

# Operator-provided X-View HOME pose, converted from degrees and expressed in
# the revised hardware-aligned arm convention.  The values are kept at the
# precision used by the existing MoveJ implementation.
H1_ARM_HOME_Q = np.array(
    [
        -1.496550020,
        -0.761539513,
        0.203383218,
        -1.487247416,
        -2.828934371,
        -0.727837205,
        2.789105958,
        1.496550020,
        -0.761539513,
        -0.203383218,
        -1.487247416,
        2.828934371,
        -0.727837205,
        -2.789105958,
    ],
    dtype=np.float64,
)


__all__ = [
    "H1_ARM_COORDINATE_CONVENTION",
    "H1_ARM_HARD_LOWER",
    "H1_ARM_HARD_UPPER",
    "H1_ARM_HOME_Q",
    "H1_ARM_HW_TO_MODEL_SIGN",
    "H1_ARM_SAFE_LOWER",
    "H1_ARM_SAFE_UPPER",
    "H1_ARM_SAFETY_MARGIN_RAD",
    "H1_MODEL_REVISION",
    "H1_MUJOCO_URDF_FILENAME",
    "H1_LEGACY_ARM_COORDINATE_CONVENTION",
    "H1_LEGACY_ARM_TO_CURRENT_SIGN",
    "arm_positions_to_current",
]
