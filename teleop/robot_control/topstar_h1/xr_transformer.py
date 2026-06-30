"""TOPSTAR_H1 XR 变换器 — TOPSTAR 约定 + 肩部缩放 + reach 限幅 + 头部 IK。"""
import numpy as np
from teleop.robot_control._base.xr_transformer import (
    XRTransformer, XRProcessedData, fast_mat_inv
)

_T_ROBOT_OPENXR_H1 = np.array([[ 0, 0, 1, 0],
                                [ 1, 0, 0, 0],
                                [ 0, 1, 0, 0],
                                [ 0, 0, 0, 1]])

# Hand tracking 路径：OpenXR arm convention → TOPSTAR arm convention
_T_LEFT_ARM_H1  = np.array([[1,0,0,0],[0,0,-1,0],[0,1,0,0],[0,0,0,1]])
_T_RIGHT_ARM_H1 = np.array([[1,0,0,0],[0,0,1,0],[0,-1,0,0],[0,0,0,1]])

# Controller 路径：TOPSTAR_H1 特有变换
_T_LEFT_CTRL_H1  = np.array([[ 0, 0, -1, 0],
                              [ 0, -1,  0, 0],
                              [-1,  0,  0, 0],
                              [ 0,  0,  0, 1]])
_T_RIGHT_CTRL_H1 = np.array([[ 0, 0, -1, 0],
                              [ 0, -1,  0, 0],
                              [-1,  0,  0, 0],
                              [ 0,  0,  0, 1]])
_T_HEAD_CTRL_H1 = np.array([[1, 0, 0, 0],
                             [0, 0, -1, 0],
                             [0, 1, 0, 0],
                             [0, 0, 0, 1]])


class H1XRTransformer(XRTransformer):

    def __init__(self, arm_ik, arm_scale=1.0, max_reach=0.69):
        super().__init__(arm_ik, arm_scale)
        self.max_reach = max_reach
        self.l_shoulder, self.r_shoulder = arm_ik.get_shoulder_positions()
        self._init_head_robot = None

    @property
    def T_ROBOT_OPENXR(self): return _T_ROBOT_OPENXR_H1
    @property
    def T_LEFT_ARM(self): return _T_LEFT_ARM_H1
    @property
    def T_RIGHT_ARM(self): return _T_RIGHT_ARM_H1
    @property
    def T_LEFT_ARM_CTRL(self): return _T_LEFT_CTRL_H1
    @property
    def T_RIGHT_ARM_CTRL(self): return _T_RIGHT_CTRL_H1
    @property
    def T_HEAD_CTRL(self): return _T_HEAD_CTRL_H1

    # ── 相对位姿模式：相似变换矩阵 ──────────────────────────────

    @property
    def S_LEFT_HAND_TO_EE(self) -> np.ndarray:
        return np.array([[ 0, 0, -1, 0],
                          [ 0, -1,  0, 0],
                          [-1,  0,  0, 0],
                          [ 0,  0,  0, 1]])

    @property
    def S_RIGHT_HAND_TO_EE(self) -> np.ndarray:
        return np.array([[ 0, 0, -1, 0],
                          [ 0, -1,  0, 0],
                          [-1,  0,  0, 0],
                          [ 0,  0,  0, 1]])

    @property
    def S_LEFT_HAND_TO_RIGHT_EE(self) -> np.ndarray:
        return np.array([[ 0, 0, -1, 0],
                          [ 0, 1,  0, 0],
                          [-1,  0,  0, 0],
                          [ 0,  0,  0, 1]])

    @property
    def S_RIGHT_HAND_TO_LEFT_EE(self) -> np.ndarray:
        return np.array([[ 0, 0, -1, 0],
                          [ 0, 1,  0, 0],
                          [-1,  0,  0, 0],
                          [ 0,  0,  0, 1]])

    def _transform_head(self, tele_data, init_head, head_is_valid, head_pose,
                        hand_tracking=True):
        """头部相对于初始头部位姿，用于 IK 优化。"""
        if init_head is not None and self._init_head_robot is None:
            self._init_head_robot = self._xr_to_robot(init_head)
        if self._init_head_robot is not None:
            head_robot = self._xr_to_robot(head_pose)
            relative = fast_mat_inv(self._init_head_robot) @ head_robot
            relative[0, 3] -= 0.3
            relative[2, 3] += 1.2
            # 控制器路径需要额外的 head 变换
            if not hand_tracking:
                relative = relative @ self.T_HEAD_CTRL
            return {"head_target": relative}
        return {"head_target": None}

    def _preprocess_arm_poses(self, left, right, hand_tracking=True):
        # 控制器路径：腰部位移（head→waist）——必须在肩部缩放之前
        if not hand_tracking:
            left[:3, 3]  += [-0.3, 0, 1.2]
            right[:3, 3] += [-0.3, 0, 1.2]
        s = self.arm_scale
        left[:3,3]  = self.l_shoulder + s * (left[:3,3] - self.l_shoulder)
        right[:3,3] = self.r_shoulder + s * (right[:3,3] - self.r_shoulder)
        for wrist, shoulder in [(left, self.l_shoulder), (right, self.r_shoulder)]:
            v = wrist[:3,3] - shoulder
            d = np.linalg.norm(v)
            if d > self.max_reach:
                v *= self.max_reach / d
                wrist[:3,3] = shoulder + v
        return left, right

    def reset_init_head(self):
        self._init_head_robot = None
