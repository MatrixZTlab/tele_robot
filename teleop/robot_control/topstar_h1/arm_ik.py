"""TOPSTAR_H1 — 逆运动学求解 (H1ArmIK)。

从现有 topstar_h1_sim_arm_ik.py 迁移，继承 BaseArmIK。
"""
import os
import sys

import numpy as np
import pinocchio as pin
from pinocchio import casadi as cpin
import casadi
from pinocchio.visualize import MeshcatVisualizer
import meshcat.geometry as mg

from teleop.robot_control._base.arm_ik import BaseArmIK, IKResult
from teleop.robot_control._base.control_mode import ControlMode
from teleop.utils.weighted_moving_filter import WeightedMovingFilter


class H1ArmIK(BaseArmIK):
    """TOPSTAR_H1 双臂 + 头部 IK 求解器。"""

    def __init__(self, config, control_mode=ControlMode.ARMS_HEAD,
                 visualization="off", verbose=False):
        # 缓存路径用本文件同级目录
        self._self_dir = os.path.dirname(os.path.abspath(__file__))
        config.cache_filename = os.path.join(self._self_dir, "topstar_h1_model_cache.pkl")

        super().__init__(config, control_mode, visualization, verbose)

    # ── 构建缩减模型 ────────────────────────────────────────

    def _build_reduced_model(self):
        pin = self._get_pin()

        robot = pin.RobotWrapper.BuildFromURDF(
            self.config.urdf_path, self.config.model_dir
        )

        # 根据 control_mode 决定锁哪些关节
        locked = list(self.config.locked_joint_names)
        if not self.control_mode.solve_head:
            locked.extend([
                "Robot_Head_Rotation_Joint",
                "Robot_Head_Tonod_Joint",
            ])

        locked_ids = [
            robot.model.getJointId(name)
            for name in locked
            if robot.model.existJointName(name)
        ]
        q0 = pin.neutral(robot.model)
        reduced_robot = robot.buildReducedRobot(locked_ids, q0)
        return robot, reduced_robot

    # ── 限位覆盖 ────────────────────────────────────────────

    def _setup_limits(self):
        if self.config.limit_mode == "modified":
            self._apply_joint_specs_limits()

    def _apply_joint_specs_limits(self):
        """覆盖 URDF 限位，使用 JOINT_SPECS 硬件限位 + 5° 安全裕度。"""
        SAFETY_DEG = 5.0
        margin = np.deg2rad(SAFETY_DEG)

        _HW_SPECS = [
            (-2.61799388,  2.61799388),   # sim[0]  L Shoulder Base
            (-1.57079633,  0.43633231),   # sim[1]  L Shoulder
            (-2.61799388,  2.61799388),   # sim[2]  L Elbow Yaw
            (-1.79768913,  0.43633231),   # sim[3]  L Elbow
            (-2.87979327,  2.87979327),   # sim[4]  L Wrist Yaw
            (-1.53588974,  0.43633231),   # sim[5]  L Wrist Pitch
            (-2.96705973,  2.96705973),   # sim[6]  L Wrist Roll
            (-2.61799388,  2.61799388),   # sim[7]  R Shoulder Base
            (-1.57079633,  0.43633231),   # sim[8]  R Shoulder
            (-2.61799388,  2.61799388),   # sim[9]  R Elbow Yaw
            (-1.79768913,  0.43633231),   # sim[10] R Elbow
            (-2.87979327,  2.87979327),   # sim[11] R Wrist Yaw
            (-1.53588974,  0.43633231),   # sim[12] R Wrist Pitch
            (-2.96705973,  2.96705973),   # sim[13] R Wrist Roll
        ]
        _HW_TO_SIM_SIGN = np.array([
             1.0,  1.0,  1.0, -1.0,  1.0, -1.0,  1.0,
             1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,
        ])

        model = self.reduced_robot.model
        for i, idx in enumerate(self.arm_joint_indices):
            if i >= len(_HW_SPECS):
                break
            hw_lo, hw_hi = _HW_SPECS[i]
            sign = _HW_TO_SIM_SIGN[i]

            if sign > 0:
                sim_lo, sim_hi = hw_lo, hw_hi
            else:
                sim_lo = hw_hi / sign
                sim_hi = hw_lo / sign

            lo_safe = sim_lo + margin
            hi_safe = sim_hi - margin
            if lo_safe < hi_safe:
                model.lowerPositionLimit[idx] = lo_safe
                model.upperPositionLimit[idx] = hi_safe

    # ── 可视化 ──────────────────────────────────────────────

    def _setup_visualization(self):
        if not self.Visualization:
            return

        self.vis = MeshcatVisualizer(
            self.reduced_robot.model,
            self.reduced_robot.collision_model,
            self.reduced_robot.visual_model,
        )
        self.vis.initViewer(open=True)
        self.vis.loadViewerModel("pinocchio")

        frame_ids = [self.L_hand_id, self.R_hand_id]
        if self.head_frame_id is not None:
            frame_ids.append(self.head_frame_id)
        self.vis.displayFrames(True, frame_ids=frame_ids, axis_length=0.15, axis_width=5)
        self.vis.display(pin.neutral(self.reduced_robot.model))

        # 目标帧可视化
        frame_viz_names = ["L_ee_target", "R_ee_target", "Head_frame", "Head_target"]
        FRAME_AXIS_POSITIONS = np.array([
            [0, 0, 0], [1, 0, 0], [0, 0, 0], [0, 1, 0], [0, 0, 0], [0, 0, 1],
        ], dtype=np.float32).T
        FRAME_AXIS_COLORS = np.array([
            [1, 0, 0], [1, 0.6, 0], [0, 1, 0], [0.6, 1, 0], [0, 0, 1], [0, 0.6, 1],
        ], dtype=np.float32).T
        axis_length = 0.1
        axis_width = 20

        for name in frame_viz_names:
            self.vis.viewer[name].set_object(
                mg.LineSegments(
                    mg.PointsGeometry(
                        position=axis_length * FRAME_AXIS_POSITIONS,
                        color=FRAME_AXIS_COLORS,
                    ),
                    mg.LineBasicMaterial(linewidth=axis_width, vertexColors=True),
                )
            )

    # ── 肩部位置（H1 特有） ─────────────────────────────────

    def get_shoulder_positions(self, q=None):
        """返回双肩位置 (l_xyz, r_xyz)。"""
        if q is None:
            q = pin.neutral(self.reduced_robot.model)
        q = np.asarray(q).reshape(-1)
        pin.forwardKinematics(self.reduced_robot.model, self.reduced_robot.data, q)
        pin.updateFramePlacements(self.reduced_robot.model, self.reduced_robot.data)

        l_shoulder_id = self.reduced_robot.model.getFrameId("Robot_Left_Hand_base_Link")
        r_shoulder_id = self.reduced_robot.model.getFrameId("Robot_Right_Hand_base_Link")
        l = self.reduced_robot.data.oMf[l_shoulder_id].translation.copy()
        r = self.reduced_robot.data.oMf[r_shoulder_id].translation.copy()
        return l, r

    # ── solve_ik（覆写以支持 H1 特有的头部处理 + 可视化） ──

    def solve_ik(self, left_wrist_pose, right_wrist_pose,
                 current_lr_arm_q=None, current_lr_arm_dq=None,
                 head_target=None, ros2_published_arm_q=None):
        """H1 特有：处理头部 IK 优化 + 可视化三模式。"""
        # ── 初始化 init_q ──
        init_q = np.array(self.init_data).reshape(-1)
        if init_q.size != self.reduced_robot.model.nq:
            init_q = np.zeros(self.reduced_robot.model.nq)

        if current_lr_arm_q is not None:
            cur = np.asarray(current_lr_arm_q).reshape(-1)
            for i, idx in enumerate(self.arm_joint_indices):
                if i < cur.size and idx < init_q.size:
                    init_q[idx] = cur[i]

        self.opti.set_value(self.param_tf_l, casadi.DM(left_wrist_pose))
        self.opti.set_value(self.param_tf_r, casadi.DM(right_wrist_pose))

        # ── 可视化：目标帧 ──
        if self.Visualization:
            self.vis.viewer["L_ee_target"].set_transform(left_wrist_pose)
            self.vis.viewer["R_ee_target"].set_transform(right_wrist_pose)
            try:
                if self.head_frame_id is not None:
                    pin.forwardKinematics(self.reduced_robot.model, self.reduced_robot.data, init_q)
                    pin.updateFramePlacements(self.reduced_robot.model, self.reduced_robot.data)
                    head_tf = self.reduced_robot.data.oMf[self.head_frame_id].homogeneous
                    self.vis.viewer["Head_frame"].set_transform(head_tf)
                if head_target is not None:
                    self.vis.viewer["Head_target"].set_transform(head_target)
            except Exception:
                pass

        # ── 头部处理 ──
        head_q_target = None
        if (self.control_mode.solve_head
                and self.head_rotation_error is not None
                and self.param_tf_head is not None):
            if head_target is not None:
                try:
                    self.opti.set_value(self.param_tf_head, casadi.DM(head_target))
                except Exception:
                    pass
            elif self.last_head_q is not None and self.last_head_q.size > 0 and self.head_frame_id is not None:
                q_tmp = np.array(self.init_data).reshape(-1)
                offset = 0
                for _, _, idx, nq in self.head_joint_map:
                    q_tmp[idx: idx + nq] = self.last_head_q[offset: offset + nq]
                    offset += nq
                try:
                    pin.forwardKinematics(self.reduced_robot.model, self.reduced_robot.data, q_tmp)
                    pin.updateFramePlacements(self.reduced_robot.model, self.reduced_robot.data)
                    tf = self.reduced_robot.data.oMf[self.head_frame_id].homogeneous
                    self.opti.set_value(self.param_tf_head, casadi.DM(tf))
                except Exception:
                    pass

            # 头部初值
            if len(self.head_joint_map) > 0 and self.last_head_q is not None:
                offset = 0
                for _, _, idx, nq in self.head_joint_map:
                    init_q[idx: idx + nq] = self.last_head_q[offset: offset + nq]
                    offset += nq

        self.init_data = init_q
        self.opti.set_initial(self.var_q, self.init_data)
        self.opti.set_value(self.var_q_last, self.init_data)

        # ── 求解 ──
        try:
            sol = self.opti.solve()
            sol_q_raw = np.array(sol.value(self.var_q)).reshape(-1)

            # 臂部平滑
            sol_q_arm_raw = sol_q_raw[self.arm_joint_indices]
            self.smooth_filter.add_data(sol_q_arm_raw)
            sol_q_arm = np.array(self.smooth_filter.filtered_data)

            # 头部提取
            if (self.control_mode.solve_head
                    and len(self.head_joint_map) > 0):
                head_list = []
                for _, _, idx, nq in self.head_joint_map:
                    head_list.append(sol_q_raw[idx: idx + nq])
                head_q_target = np.concatenate(head_list) if head_list else None
                if head_q_target is not None:
                    self.last_head_q = head_q_target.copy()

            # 力矩
            v_full = np.zeros(self.reduced_robot.model.nv)
            tau_full = pin.rnea(
                self.reduced_robot.model, self.reduced_robot.data,
                sol_q_raw, v_full, np.zeros(self.reduced_robot.model.nv),
            )
            self.init_data = sol_q_raw.copy()

            # 可视化
            if self.Visualization:
                try:
                    q_vis = self._build_visual_q(sol_q_raw, current_lr_arm_q, ros2_published_arm_q)
                    self.vis.display(q_vis)
                    if self.head_frame_id is not None:
                        pin.forwardKinematics(self.reduced_robot.model, self.reduced_robot.data, sol_q_raw)
                        pin.updateFramePlacements(self.reduced_robot.model, self.reduced_robot.data)
                        self.vis.viewer["Head_frame"].set_transform(
                            self.reduced_robot.data.oMf[self.head_frame_id].homogeneous
                        )
                except Exception:
                    pass

            return IKResult(
                arm_q=sol_q_arm,
                arm_tau=np.array(tau_full)[self.arm_joint_indices],
                head_q=head_q_target,
            )

        except Exception as e:
            if self.verbose:
                print(f"[H1ArmIK] solve failed: {e}")

            sol_q_raw = np.array(self.opti.debug.value(self.var_q)).reshape(-1)
            sol_q_arm_raw = sol_q_raw[self.arm_joint_indices]
            self.smooth_filter.add_data(sol_q_arm_raw)
            sol_q_arm = np.array(self.smooth_filter.filtered_data)

            head_q_target = None
            if self.control_mode.solve_head and len(self.head_joint_map) > 0:
                try:
                    head_list = []
                    for _, _, idx, nq in self.head_joint_map:
                        head_list.append(sol_q_raw[idx: idx + nq])
                    head_q_target = np.concatenate(head_list) if head_list else None
                except Exception:
                    head_q_target = self.last_head_q.copy() if self.last_head_q is not None else None

            if current_lr_arm_q is not None:
                return IKResult(
                    arm_q=np.array(current_lr_arm_q).reshape(-1)[:14],
                    arm_tau=np.zeros(14),
                    head_q=head_q_target,
                )
            return IKResult(
                arm_q=sol_q_arm,
                arm_tau=np.zeros(len(sol_q_arm)),
                head_q=head_q_target,
            )

    def _build_visual_q(self, sol_q_raw, current_lr_arm_q, ros2_published_arm_q):
        """可视化模式：ik=原始求解结果，ros2=实际下发命令。"""
        if self.visualization_mode == "ik":
            return np.array(sol_q_raw).reshape(-1)

        q_vis = np.array(sol_q_raw).reshape(-1).copy()
        if ros2_published_arm_q is not None:
            q_cmd_arm = np.array(ros2_published_arm_q).reshape(-1)[:14]
        elif current_lr_arm_q is not None:
            q_cmd_arm = np.array(current_lr_arm_q).reshape(-1)[:14]
        else:
            q_cmd_arm = None

        if q_cmd_arm is None or q_cmd_arm.size == 0:
            return q_vis
        for i, idx in enumerate(self.arm_joint_indices):
            if i < q_cmd_arm.size and idx < q_vis.size:
                q_vis[idx] = q_cmd_arm[i]
        return q_vis

    @staticmethod
    def _get_pin():
        return pin
