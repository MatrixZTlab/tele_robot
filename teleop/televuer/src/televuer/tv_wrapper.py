import numpy as np
from .televuer import TeleVuer
from dataclasses import dataclass, field
from typing import Literal
from teleop.robot_control._base.xr_transformer import (
    safe_mat_update, safe_rot_update, fast_mat_inv,
)

"""
(basis) OpenXR Convention : y up, z back, x right. 
(basis) Robot  Convention : z up, y left, x front.  

under (basis) Robot Convention, humanoid arm's initial pose convention:

    # (initial pose) OpenXR Left Arm Pose Convention (hand tracking):
        - the x-axis pointing from wrist toward middle.
        - the y-axis pointing from index toward pinky.
        - the z-axis pointing from palm toward back of the hand.

    # (initial pose) OpenXR Right Arm Pose Convention (hand tracking):
        - the x-axis pointing from wrist toward middle.
        - the y-axis pointing from pinky toward index.
        - the z-axis pointing from palm toward back of the hand.
  
    # (initial pose) TOPSTAR Humanoid Left Arm URDF Convention:
        - the x-axis pointing from wrist toward middle.
        - the y-axis pointing from palm toward back of the hand.
        - the z-axis pointing from pinky toward index.

    # (initial pose) TOPSTAR Humanoid Right Arm URDF Convention:
        - the x-axis pointing from wrist toward middle.
        - the y-axis pointing from back of the hand toward palm. 
        - the z-axis pointing from pinky toward index.

under (basis) Robot Convention, humanoid hand's initial pose convention:

    # (initial pose) OpenXR Left Hand Pose Convention (hand tracking):
        - the x-axis pointing from wrist toward middle.
        - the y-axis pointing from index toward pinky.
        - the z-axis pointing from palm toward back of the hand.

    # (initial pose) OpenXR Right Hand Pose Convention (hand tracking):
        - the x-axis pointing from wrist toward middle.
        - the y-axis pointing from pinky toward index.
        - the z-axis pointing from palm toward back of the hand.

    # (initial pose) TOPSTAR Humanoid Left Hand URDF Convention:
        - The x-axis pointing from palm toward back of the hand. 
        - The y-axis pointing from middle toward wrist.
        - The z-axis pointing from pinky toward index.

    # (initial pose) TOPSTAR Humanoid Right Hand URDF Convention:
        - The x-axis pointing from palm toward back of the hand. 
        - The y-axis pointing from middle toward wrist.
        - The z-axis pointing from index toward pinky. 
    
p.s. TeleVuer obtains all raw data under the (basis) OpenXR Convention. 
     In addition, arm pose data (hand tracking) follows the (initial pose) OpenXR Arm Pose Convention, 
     while arm pose data (controller tracking) directly follows the (initial pose) TOPSTAR Humanoid Arm URDF Convention (thus no transform is needed).
     Meanwhile, all raw data is in the WORLD frame defined by XR device odometry.

p.s. From website: https://registry.khronos.org/OpenXR/specs/1.1/man/html/openxr.html.
     You can find **(initial pose) OpenXR Left/Right Arm Pose Convention** related information like this below:
     "The wrist joint is located at the pivot point of the wrist, which is location invariant when twisting the hand without moving the forearm. 
     The backward (+Z) direction is parallel to the line from wrist joint to middle finger metacarpal joint, and points away from the finger tips. 
     The up (+Y) direction points out towards back of the hand and perpendicular to the skin at wrist. 
     The X direction is perpendicular to the Y and Z directions and follows the right hand rule."
     Note: The above context is of course under **(basis) OpenXR Convention**.

p.s. **TOPSTAR Arm/Hand URDF initial pose Convention** information come from URDF files.
"""


# 所有变换矩阵已移至 XRTransformer 子类，此处仅保留原始 OpenXR 数据。

CONST_HEAD_POSE = np.array([[1, 0, 0, 0],
                            [0, 1, 0, 1.5],
                            [0, 0, 1, -0.2],
                            [0, 0, 0, 1]])

CONST_LEFT_ARM_POSE = np.array([[1, 0, 0, -0.15],
                                [0, 1, 0, 1.13],
                                [0, 0, 1, -0.3],
                                [0, 0, 0, 1]])

CONST_RIGHT_ARM_POSE = np.array([[1, 0, 0, 0.15],
                                 [0, 1, 0, 1.13],
                                 [0, 0, 1, -0.3],
                                 [0, 0, 0, 1]])

CONST_HAND_ROT = np.tile(np.eye(3)[None, :, :], (25, 1, 1))

@dataclass
class TeleData:
    head_pose: np.ndarray                  # (4,4) SE(3) pose, OpenXR convention, raw
    left_wrist_pose: np.ndarray            # (4,4) SE(3) pose, OpenXR convention, raw
    right_wrist_pose: np.ndarray           # (4,4) SE(3) pose, OpenXR convention, raw
    left_wrist_valid: bool = False         # False means wrist_pose is the safe fallback
    right_wrist_valid: bool = False        # False means wrist_pose is the safe fallback
    controller_pose_timestamp_ns: int = 0  # monotonic clock, updated on WebSocket input
    controller_pose_wall_ns: int = 0       # host wall clock at WebSocket receipt
    controller_pose_sequence: int = 0
    controller_source_timestamp_raw: float = 0.0
    # hand tracking
    left_hand_pos: np.ndarray = None       # (25,3) OpenXR convention, raw
    right_hand_pos: np.ndarray = None      # (25,3) OpenXR convention, raw
    left_hand_rot: np.ndarray  = None      # (25,3,3) OpenXR convention, raw
    right_hand_rot: np.ndarray = None      # (25,3,3) OpenXR convention, raw
    # init head pose (raw OpenXR, frozen on first valid frame after capture_init_head_pose)
    init_head_pose: np.ndarray = None      # (4,4) OpenXR convention, raw
    use_hand_tracking: bool = True
    # teleop mode
    teleop_mode: str = "relative_head"     # "relative_head" | "relative_pose"
    # wrist reference (raw OpenXR, frozen on A-button press in relative_pose mode)
    left_wrist_ref: np.ndarray = None      # (4,4) OpenXR convention, raw
    right_wrist_ref: np.ndarray = None     # (4,4) OpenXR convention, raw
    # HandState
    left_hand_pinch: bool = False          # True if index and thumb are pinching
    left_hand_pinchValue: float = 10.0     # float (~15.0​​ → 0.0) pinch distance between index and thumb
    left_hand_squeeze: bool = False        # True if hand is making a fist
    left_hand_squeezeValue: float = 0.0    # (0.0 → 1.0) degree of hand squeeze

    right_hand_pinch: bool = False         # True if index and thumb are pinching
    right_hand_pinchValue: float = 10.0    # float (~15.0​​ → 0.0) pinch distance between index and thumb
    right_hand_squeeze: bool = False       # True if hand is making a fist
    right_hand_squeezeValue: float = 0.0   # (0.0 → 1.0) degree of hand squeeze

    # controller tracking
    # https://docs.vuer.ai/en/latest/examples/20_motion_controllers.html
    # https://immersive-web.github.io/webxr-gamepads-module/
    left_ctrl_trigger: bool = False        # True if trigger is actively pressed
    left_ctrl_triggerValue: float = 10.0   # float (10.0 → 0.0) trigger pull depth, 0.0 means fully pressed (for align with hand pinch value's logic)
    left_ctrl_squeeze: bool = False        # True if grip button is pressed
    left_ctrl_squeezeValue: float = 0.0    # (0.0 → 1.0) grip pull depth, 0.0 means no press
    left_ctrl_aButton: bool = False        # True if A(X) button is pressed
    left_ctrl_bButton: bool = False        # True if B(Y) button is pressed
    left_ctrl_thumbstick: bool = False     # True if thumbstick button is pressed
    left_ctrl_thumbstickValue: np.ndarray = field(default_factory=lambda: np.zeros(2)) # 2D vector (x, y), normalized
    """ thumbstickValue explanation:
                    front (0,-1)
                       ^
                       |
      left (-1,0) < —— o —— > right (1,0)      and 'o' is at (0, 0)
                       |
                       v
                    back (0,1)
    """
    right_ctrl_trigger: bool = False       # True if trigger is actively pressed
    right_ctrl_triggerValue: float = 10.0  # float (10.0 → 0.0) trigger pull depth, 0.0 means fully pressed (for align  with hand pinch value's logic)
    right_ctrl_squeeze: bool = False       # True if grip button is pressed
    right_ctrl_squeezeValue: float = 0.0   # (0.0 → 1.0) grip pull depth, 0.0 means no press
    right_ctrl_aButton: bool = False       # True if A button is pressed
    right_ctrl_bButton: bool = False       # True if B button is pressed
    right_ctrl_thumbstick: bool = False    # True if thumbstick button is pressed
    right_ctrl_thumbstickValue: np.ndarray = field(default_factory=lambda: np.zeros(2)) # 2D vector (x, y), normalized


class TeleVuerWrapper:
    def __init__(self, use_hand_tracking: bool, binocular: bool=True, img_shape: tuple=(480, 1280), display_fps: float=30.0,
                       display_mode: Literal["immersive", "pass-through", "ego"]="immersive", zmq: bool=False, webrtc: bool=False, webrtc_url: str=None, 
                       cert_file: str=None, key_file: str=None, return_hand_rot_data: bool=False):
        """
        TeleVuerWrapper is a wrapper for the TeleVuer class, which handles XR device's data suit for robot control.
        It initializes the TeleVuer instance with the specified parameters and provides a method to get motion state data.

        :param use_hand_tracking: bool, whether to use hand tracking or controller tracking.
        :param binocular: bool, whether the application is binocular (stereoscopic) or monocular.
        :param img_shape: tuple, shape of the head image (height, width).
        :param display_fps: float, target frames per second for display updates (default: 30.0).

        :param display_mode: str, controls the VR viewing mode. Options are "immersive", "pass-through", and "ego".
        :param zmq: bool, whether to use ZMQ for image transmission.
        :param webrtc: bool, whether to use webrtc for real-time communication.
        :param webrtc_url: str, URL for the webrtc offer. must be provided if webrtc is True.
        :param cert_file: str, path to the SSL certificate file.
        :param key_file: str, path to the SSL key file.

        Note:

        - display_mode controls what the VR headset displays:
            * "immersive": fully immersive mode; VR shows the robot's first-person view (zmq or webrtc must be enabled).
            * "pass-through": VR shows the real world through the VR headset cameras; no image from zmq or webrtc is displayed (even if enabled).
            * "ego": a small window in the center shows the robot's first-person view, while the surrounding area shows the real world.
        
        - Only one image mode is active at a time.
        - Image transmission to VR occurs only if display_mode is "immersive" or "ego" and the corresponding zmq or webrtc option is enabled.
        - If zmq and webrtc simultaneously enabled, webrtc will be prioritized.

        --------------              -------------------           --------------       -----------------                     -------
         display_mode       |        display behavior         |    image to VR     |      image source        |               Notes
        --------------              -------------------           --------------       -----------------                     ------- 
           immersive        |   fully immersive view (robot)  |     Yes (full)     |     zmq or webrtc        |   if both enabled, webrtc prioritized
        --------------              -------------------           --------------       -----------------                     -------
         pass-through       |       Real world view (VR)      |         No         |          N/A             |  even if image source enabled, don't display
        --------------              -------------------           --------------       -----------------                     -------
              ego           |      ego view (robot + VR)      |    Yes (small)     |     zmq or webrtc        |   if both enabled, webrtc prioritized
        --------------              -------------------           --------------       -----------------                     -------
        """
        self.use_hand_tracking = use_hand_tracking
        self.return_hand_rot_data = return_hand_rot_data
        self.tvuer = TeleVuer(use_hand_tracking=use_hand_tracking, binocular=binocular, img_shape=img_shape, display_fps=display_fps,
                              display_mode=display_mode, zmq=zmq, webrtc=webrtc, webrtc_url=webrtc_url, 
                              cert_file=cert_file, key_file=key_file)
        # Cached first valid head pose (OpenXR convention). Set once and then frozen.
        self._init_head_pose = None
        # Flag: when True, the next valid head frame in get_tele_data() will be captured as _init_head_pose.
        self._pending_init_capture = False

        # Wrist reference poses for relative_pose mode (raw OpenXR)
        self._left_wrist_ref = None
        self._right_wrist_ref = None
        self._pending_left_wrist_capture = False
        self._pending_right_wrist_capture = False

    def reset_init_head_pose(self):
        """Clear the initial head pose reference.  After this call _init_head_pose is None
        and no automatic capture will happen until ``capture_init_head_pose()`` is called."""
        self._init_head_pose = None
        self._pending_init_capture = False

    def drain_raw_xr_events(self, max_items=512):
        """Return every queued XR event, not only the latest control snapshot."""
        return self.tvuer.drain_xr_events(max_items=max_items)

    def capture_init_head_pose(self):
        """Request that the next valid head frame be frozen as the initial head reference."""
        self._init_head_pose = None
        self._pending_init_capture = True

    # ── Wrist reference management (for relative_pose mode) ────────────────

    def capture_left_wrist_ref(self):
        """Request that the next valid left wrist frame be frozen as reference."""
        self._left_wrist_ref = None
        self._pending_left_wrist_capture = True

    def capture_right_wrist_ref(self):
        """Request that the next valid right wrist frame be frozen as reference."""
        self._right_wrist_ref = None
        self._pending_right_wrist_capture = True

    def reset_left_wrist_ref(self):
        self._left_wrist_ref = None
        self._pending_left_wrist_capture = False

    def reset_right_wrist_ref(self):
        self._right_wrist_ref = None
        self._pending_right_wrist_capture = False

    def reset_wrist_refs(self):
        self.reset_left_wrist_ref()
        self.reset_right_wrist_ref()

    def get_tele_data(self):
        """
        Get raw motion state data from the TeleVuer instance.

        All pose data is returned under the OpenXR Convention (raw, no transforms).
        Robot-specific transforms (basis, initial pose, head, shoulder scaling)
        are handled by XRTransformer subclasses.
        """
        # ── Head pose ──
        head_pose, head_is_valid = safe_mat_update(CONST_HEAD_POSE, self.tvuer.head_pose)

        # Record init_head_pose on first valid head frame after capture_init_head_pose()
        if self._pending_init_capture and head_is_valid:
            self._init_head_pose = head_pose.copy()
            self._pending_init_capture = False

        # ── Wrist reference capture (for relative_pose mode) ──
        if self._pending_left_wrist_capture:
            _lw, _lv = safe_mat_update(CONST_LEFT_ARM_POSE, self.tvuer.left_arm_pose)
            if _lv:
                self._left_wrist_ref = _lw.copy()
                self._pending_left_wrist_capture = False
        if self._pending_right_wrist_capture:
            _rw, _rv = safe_mat_update(CONST_RIGHT_ARM_POSE, self.tvuer.right_arm_pose)
            if _rv:
                self._right_wrist_ref = _rw.copy()
                self._pending_right_wrist_capture = False

        # ── Hand tracking path ──
        if self.use_hand_tracking:
            left_wrist, left_valid  = safe_mat_update(CONST_LEFT_ARM_POSE, self.tvuer.left_arm_pose)
            right_wrist, right_valid = safe_mat_update(CONST_RIGHT_ARM_POSE, self.tvuer.right_arm_pose)

            # ── Hand positions (raw OpenXR) ──
            if left_valid and right_valid:
                left_hand_pos = self.tvuer.left_hand_positions.copy()      # (25,3)
                right_hand_pos = self.tvuer.right_hand_positions.copy()    # (25,3)
            else:
                left_hand_pos = np.zeros((25, 3))
                right_hand_pos = np.zeros((25, 3))

            # ── Hand rotations (raw OpenXR) ──
            if self.return_hand_rot_data:
                left_hand_rot, _  = safe_rot_update(CONST_HAND_ROT, self.tvuer.left_hand_orientations)
                right_hand_rot, _ = safe_rot_update(CONST_HAND_ROT, self.tvuer.right_hand_orientations)
            else:
                left_hand_rot = None
                right_hand_rot = None

            return TeleData(
                head_pose=head_pose,
                left_wrist_pose=left_wrist,
                right_wrist_pose=right_wrist,
                left_wrist_valid=left_valid,
                right_wrist_valid=right_valid,
                controller_pose_timestamp_ns=self.tvuer.controller_pose_timestamp_ns,
                controller_pose_wall_ns=self.tvuer.controller_pose_wall_ns,
                controller_pose_sequence=self.tvuer.controller_pose_sequence,
                controller_source_timestamp_raw=self.tvuer.controller_source_timestamp_raw,
                left_hand_pos=left_hand_pos,
                right_hand_pos=right_hand_pos,
                left_hand_rot=left_hand_rot,
                right_hand_rot=right_hand_rot,
                left_hand_pinch=self.tvuer.left_hand_pinch,
                left_hand_pinchValue=self.tvuer.left_hand_pinchValue * 100.0,
                left_hand_squeeze=self.tvuer.left_hand_squeeze,
                left_hand_squeezeValue=self.tvuer.left_hand_squeezeValue,
                right_hand_pinch=self.tvuer.right_hand_pinch,
                right_hand_pinchValue=self.tvuer.right_hand_pinchValue * 100.0,
                right_hand_squeeze=self.tvuer.right_hand_squeeze,
                right_hand_squeezeValue=self.tvuer.right_hand_squeezeValue,
                init_head_pose=self._init_head_pose,
                use_hand_tracking=True,
                teleop_mode="relative_head",
                left_wrist_ref=self._left_wrist_ref,
                right_wrist_ref=self._right_wrist_ref,
            )

        # ── Controller tracking path ──
        else:
            left_wrist, left_valid  = safe_mat_update(CONST_LEFT_ARM_POSE, self.tvuer.left_arm_pose)
            right_wrist, right_valid = safe_mat_update(CONST_RIGHT_ARM_POSE, self.tvuer.right_arm_pose)

            return TeleData(
                head_pose=head_pose,
                left_wrist_pose=left_wrist,
                right_wrist_pose=right_wrist,
                left_wrist_valid=left_valid,
                right_wrist_valid=right_valid,
                controller_pose_timestamp_ns=self.tvuer.controller_pose_timestamp_ns,
                controller_pose_wall_ns=self.tvuer.controller_pose_wall_ns,
                controller_pose_sequence=self.tvuer.controller_pose_sequence,
                controller_source_timestamp_raw=self.tvuer.controller_source_timestamp_raw,
                left_ctrl_trigger=self.tvuer.left_ctrl_trigger,
                left_ctrl_triggerValue=10.0 - self.tvuer.left_ctrl_triggerValue * 10,
                left_ctrl_squeeze=self.tvuer.left_ctrl_squeeze,
                left_ctrl_squeezeValue=self.tvuer.left_ctrl_squeezeValue,
                left_ctrl_aButton=self.tvuer.left_ctrl_aButton,
                left_ctrl_bButton=self.tvuer.left_ctrl_bButton,
                left_ctrl_thumbstick=self.tvuer.left_ctrl_thumbstick,
                left_ctrl_thumbstickValue=self.tvuer.left_ctrl_thumbstickValue,
                right_ctrl_trigger=self.tvuer.right_ctrl_trigger,
                right_ctrl_triggerValue=10.0 - self.tvuer.right_ctrl_triggerValue * 10,
                right_ctrl_squeeze=self.tvuer.right_ctrl_squeeze,
                right_ctrl_squeezeValue=self.tvuer.right_ctrl_squeezeValue,
                right_ctrl_aButton=self.tvuer.right_ctrl_aButton,
                right_ctrl_bButton=self.tvuer.right_ctrl_bButton,
                right_ctrl_thumbstick=self.tvuer.right_ctrl_thumbstick,
                right_ctrl_thumbstickValue=self.tvuer.right_ctrl_thumbstickValue,
                init_head_pose=self._init_head_pose,
                use_hand_tracking=False,
                teleop_mode="relative_head",
                left_wrist_ref=self._left_wrist_ref,
                right_wrist_ref=self._right_wrist_ref,
            )

        
    def render_to_xr(self, img):
        self.tvuer.render_to_xr(img)
    
    def close(self):
        self.tvuer.close()
