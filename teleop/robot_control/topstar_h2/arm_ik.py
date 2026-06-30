"""TOPSTAR_H2 逆运动学求解 — 基于 TH010 URDF，继承 BaseArmIK。"""
import os
import numpy as np
import pinocchio as pin
import casadi
from pinocchio.visualize import MeshcatVisualizer
import meshcat.geometry as mg

from teleop.robot_control._base.arm_ik import BaseArmIK, IKResult
from teleop.robot_control._base.control_mode import ControlMode


class H2ArmIK(BaseArmIK):
    """TOPSTAR_H2 双臂 + 头部 IK 求解器。"""

    def __init__(self, config, control_mode=ControlMode.ARMS_HEAD,
                 visualization="off", verbose=False):
        self._self_dir = os.path.dirname(os.path.abspath(__file__))
        config.cache_filename = os.path.join(self._self_dir, "topstar_h2_model_cache.pkl")

        super().__init__(config, control_mode, visualization, verbose)

    # ── 构建缩减模型 ────────────────────────────────────────

    def _build_reduced_model(self):
        robot = pin.RobotWrapper.BuildFromURDF(
            self.config.urdf_path, self.config.model_dir
        )

        locked = list(self.config.locked_joint_names)  # 12腿关节+腰关节
        if not self.control_mode.solve_head:
            locked.extend(["head_yaw_joint", "head_pitch_joint"])

        locked_ids = [
            robot.model.getJointId(name)
            for name in locked
            if robot.model.existJointName(name)
        ]
        q0 = pin.neutral(robot.model)
        reduced_robot = robot.buildReducedRobot(locked_ids, q0)

        # 添加末端执行器 frame
        model = reduced_robot.model
        ee_offset = np.array([0, 0, -0.1])
        for side, joint_name, frame_name in [
            ("L", "left_wrist_roll_joint", "L_ee"),
            ("R", "right_wrist_roll_joint", "R_ee"),
        ]:
            jid = model.getJointId(joint_name)
            model.addFrame(pin.Frame(
                frame_name, jid, pin.SE3(np.eye(3), ee_offset),
                pin.FrameType.OP_FRAME
            ))

        return robot, reduced_robot

    # ── JOINT_SPECS 限位（同 H1 模式，无 sign_map）────────

    def _setup_limits(self):
        if self.config.limit_mode == "modified":
            self._apply_joint_specs_limits()

    def _apply_joint_specs_limits(self):
        """覆盖 URDF 限位，使用 h2_isaac_jog.py 中的 JOINT_SPECS 值。"""
        SAFETY_DEG = 5.0
        margin = np.deg2rad(SAFETY_DEG)

        _HW_SPECS = [
            (-3.927,      1.8326),     # sim[0]  L ShoulderPitch
            (-0.15708,    3.0194),     # sim[1]  L ShoulderRoll
            (-2.9671,     2.9671),     # sim[2]  L ShoulderYaw
            (-2.2515,     1.3614),     # sim[3]  L Elbow
            (-2.9671,     2.9671),     # sim[4]  L WristYaw
            (-1.6581,     1.6581),     # sim[5]  L WristPitch
            (-1.7453,     1.7453),     # sim[6]  L WristRoll
            (-3.927,      1.8326),     # sim[7]  R ShoulderPitch
            (-3.0194,     0.15708),    # sim[8]  R ShoulderRoll
            (-2.9671,     2.9671),     # sim[9]  R ShoulderYaw
            (-1.3614,     2.2515),     # sim[10] R Elbow
            (-2.9671,     2.9671),     # sim[11] R WristYaw
            (-1.6581,     1.6581),     # sim[12] R WristPitch
            (-1.7453,     1.7453),     # sim[13] R WristRoll
        ]
        _HW_TO_SIM_SIGN = np.ones(14, dtype=float)  # H2 无 sign_map

        model = self.reduced_robot.model
        for i, idx in enumerate(self.arm_joint_indices):
            if i >= len(_HW_SPECS):
                break
            hw_lo, hw_hi = _HW_SPECS[i]
            sign = _HW_TO_SIM_SIGN[i]
            sim_lo = hw_lo if sign > 0 else hw_hi / sign
            sim_hi = hw_hi if sign > 0 else hw_lo / sign
            lo_safe = sim_lo + margin
            hi_safe = sim_hi - margin
            if lo_safe < hi_safe:
                model.lowerPositionLimit[idx] = lo_safe
                model.upperPositionLimit[idx] = hi_safe

    # ── 肩部位置 ────────────────────────────────────────────

    def get_shoulder_positions(self, q=None):
        """返回双肩位置 (l_xyz, r_xyz)。"""
        if q is None:
            q = pin.neutral(self.reduced_robot.model)
        q = np.asarray(q).reshape(-1)
        pin.forwardKinematics(self.reduced_robot.model, self.reduced_robot.data, q)
        pin.updateFramePlacements(self.reduced_robot.model, self.reduced_robot.data)

        l_id = self.reduced_robot.model.getFrameId("left_shoulder_pitch_link")
        r_id = self.reduced_robot.model.getFrameId("right_shoulder_pitch_link")
        l = self.reduced_robot.data.oMf[l_id].translation.copy()
        r = self.reduced_robot.data.oMf[r_id].translation.copy()
        return l, r

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

        frame_viz_names = ["L_ee_target", "R_ee_target", "Head_frame", "Head_target", "L_ee", "R_ee"]
        FRAME_AXIS_POSITIONS = np.array([
            [0,0,0],[1,0,0],[0,0,0],[0,1,0],[0,0,0],[0,0,1],
        ], dtype=np.float32).T
        FRAME_AXIS_COLORS = np.array([
            [1,0,0],[1,0.6,0],[0,1,0],[0.6,1,0],[0,0,1],[0,0.6,1],
        ], dtype=np.float32).T

        for name in frame_viz_names:
            self.vis.viewer[name].set_object(
                mg.LineSegments(
                    mg.PointsGeometry(
                        position=0.1 * FRAME_AXIS_POSITIONS,
                        color=FRAME_AXIS_COLORS,
                    ),
                    mg.LineBasicMaterial(linewidth=20, vertexColors=True),
                )
            )

    # ── solve_ik ────────────────────────────────────────────

    def solve_ik(self, left_wrist_pose, right_wrist_pose,
                 current_lr_arm_q=None, current_lr_arm_dq=None,
                 head_target=None, ros2_published_arm_q=None):
        """覆写：正确处理头部优化变量。"""
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

        # 可视化：目标帧
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

        # 头部处理
        head_q_target = None
        if (self.control_mode.solve_head
                and self.head_rotation_error is not None
                and self.param_tf_head is not None):
            if head_target is not None:
                self.opti.set_value(self.param_tf_head, casadi.DM(head_target))
            elif self.last_head_q is not None and self.last_head_q.size > 0 and self.head_frame_id is not None:
                q_tmp = np.array(self.init_data).reshape(-1)
                offset = 0
                for _, _, idx, nq in self.head_joint_map:
                    q_tmp[idx: idx + nq] = self.last_head_q[offset: offset + nq]
                    offset += nq
                pin.forwardKinematics(self.reduced_robot.model, self.reduced_robot.data, q_tmp)
                pin.updateFramePlacements(self.reduced_robot.model, self.reduced_robot.data)
                tf = self.reduced_robot.data.oMf[self.head_frame_id].homogeneous
                self.opti.set_value(self.param_tf_head, casadi.DM(tf))

            if len(self.head_joint_map) > 0 and self.last_head_q is not None:
                offset = 0
                for _, _, idx, nq in self.head_joint_map:
                    init_q[idx: idx + nq] = self.last_head_q[offset: offset + nq]
                    offset += nq

        self.init_data = init_q
        self.opti.set_initial(self.var_q, self.init_data)
        self.opti.set_value(self.var_q_last, self.init_data)

        try:
            sol = self.opti.solve()
            sol_q_raw = np.array(sol.value(self.var_q)).reshape(-1)

            sol_q_arm_raw = sol_q_raw[self.arm_joint_indices]
            self.smooth_filter.add_data(sol_q_arm_raw)
            sol_q_arm = np.array(self.smooth_filter.filtered_data)

            if (self.control_mode.solve_head and len(self.head_joint_map) > 0):
                head_list = []
                for _, _, idx, nq in self.head_joint_map:
                    head_list.append(sol_q_raw[idx: idx + nq])
                head_q_target = np.concatenate(head_list) if head_list else None
                if head_q_target is not None:
                    self.last_head_q = head_q_target.copy()

            v_full = np.zeros(self.reduced_robot.model.nv)
            tau_full = pin.rnea(
                self.reduced_robot.model, self.reduced_robot.data,
                sol_q_raw, v_full, np.zeros(self.reduced_robot.model.nv),
            )
            self.init_data = sol_q_raw.copy()

            # 可视化：求解后显示机器人姿态
            if self.Visualization:
                try:
                    self.vis.display(sol_q_raw)
                    pin.forwardKinematics(self.reduced_robot.model, self.reduced_robot.data, sol_q_raw)
                    pin.updateFramePlacements(self.reduced_robot.model, self.reduced_robot.data)
                    self.vis.viewer["L_ee"].set_transform(
                        self.reduced_robot.data.oMf[self.L_hand_id].homogeneous
                    )
                    self.vis.viewer["R_ee"].set_transform(
                        self.reduced_robot.data.oMf[self.R_hand_id].homogeneous
                    )
                    if self.head_frame_id is not None:
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
                print(f"[H2ArmIK] solve failed: {e}")

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
