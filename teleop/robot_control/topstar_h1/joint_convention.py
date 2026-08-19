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

# Previous operator-validated X-View HOME captured on 2026-08-10.  Keep this
# available so the trial HOME below can be reverted without reconstructing it.
H1_ARM_PREVIOUS_HOME_Q = np.array(
    [
        # Robot-left J1..J7
        -1.322488334,
        -0.800390542,
        0.287473181,
        -1.509046578,
        -2.502104016,
        -0.817320235,
        -0.526391302,
        # Robot-right J1..J7
        1.056622329,
        -0.701098761,
        0.062639867,
        -1.608774691,
        2.476866555,
        -0.845629476,
        0.410972679,
    ],
    dtype=np.float64,
)

# Retained trial X-View HOME captured on 2026-08-12.
H1_ARM_HOME_Q_2026_08_12 = np.array(
    [
        # Robot-left J1..J7
        -1.245100435080,
        -0.767246739177,
        0.243124364803,
        -1.148479007690,
        -2.212990225066,
        -0.503894008343,
        -0.953019584759,
        # Robot-right J1..J7
        0.986494999812,
        -0.585016911976,
        0.187954507147,
        -1.333972600592,
        2.182796029007,
        -0.435773807638,
        0.766408981136,
    ],
    dtype=np.float64,
)

# Preserve the complete HOME that was active before the right-arm adjustment
# captured at 11:21 on 2026-08-13.
H1_ARM_HOME_Q_BEFORE_RIGHT_ARM_2026_08_13_1121 = np.array(
    [
        # Robot-left J1..J7 (lower X-View window)
        -1.251523246728,
        -0.617270596553,
        0.290911479722,
        -1.399317727786,
        -2.319420402853,
        -0.731397676341,
        -0.829869152738,
        # Robot-right J1..J7 (upper X-View window)
        0.988833741010,
        -0.655091881444,
        0.185964831800,
        -1.372282577673,
        2.200947453227,
        -0.541331320799,
        0.749025501786,
    ],
    dtype=np.float64,
)

# Preserve the HOME that was active immediately before the complete two-arm
# X-View recapture at 16:59 on 2026-08-13.
H1_ARM_HOME_Q_BEFORE_2026_08_13_1659 = np.array(
    [
        # Robot-left J1..J7
        -1.251523246728,
        -0.617270596553,
        0.290911479722,
        -1.399317727786,
        -2.319420402853,
        -0.731397676341,
        -0.829869152738,
        # Robot-right J1..J7
        0.988694114670,
        -0.655196601199,
        0.185964831800,
        -1.372265124381,
        2.226341993844,
        -0.541313867506,
        0.712425947372,
    ],
    dtype=np.float64,
)

# Operator-selected HOME captured from X-View at 16:59 on 2026-08-13.
# Every teleoperation array is robot-left J1..J7 followed by right J1..J7.
H1_ARM_HOME_Q = np.array(
    [
        # Robot-left J1..J7: [-71.599, -25.196, 22.378, -98.153,
        #                     -131.722, -51.871, -44.426] degrees
        -1.249638291135,
        -0.439753158332,
        0.390569780011,
        -1.713093020710,
        -2.298982597312,
        -0.905319736302,
        -0.775379973491,
        # Robot-right J1..J7: [72.050, -24.118, -23.068, -97.438,
        #                      124.634, -47.599, 54.002] degrees
        1.257509726062,
        -0.420938508996,
        -0.402612551850,
        -1.700613916558,
        2.175273659931,
        -0.830759270657,
        0.942512702662,
    ],
    dtype=np.float64,
)


__all__ = [
    "H1_ARM_COORDINATE_CONVENTION",
    "H1_ARM_HARD_LOWER",
    "H1_ARM_HARD_UPPER",
    "H1_ARM_HOME_Q",
    "H1_ARM_HOME_Q_BEFORE_2026_08_13_1659",
    "H1_ARM_HOME_Q_BEFORE_RIGHT_ARM_2026_08_13_1121",
    "H1_ARM_HOME_Q_2026_08_12",
    "H1_ARM_PREVIOUS_HOME_Q",
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
