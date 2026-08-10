from __future__ import annotations

import os
import pickle
import warnings
import time
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

# ── 这些包仅在构建 IK 模型时惰性加载 ─────────────────────────
# 避免所有 import 者都被迫安装 casadi/pinocchio
_pin = None
_cpin = None
_casadi = None
_mg = None
_MeshcatVisualizer = None


def _lazy_import_pin():
    global _pin
    if _pin is None:
        import pinocchio as pin
        _pin = pin
    return _pin


def _lazy_import_cpin():
    global _cpin
    if _cpin is None:
        from pinocchio import casadi as cpin
        _cpin = cpin
    return _cpin


def _lazy_import_casadi():
    global _casadi
    if _casadi is None:
        import casadi
        _casadi = casadi
    return _casadi


def _lazy_import_meshcat():
    global _mg, _MeshcatVisualizer
    if _mg is None:
        import meshcat.geometry as mg
        from pinocchio.visualize import MeshcatVisualizer
        _mg = mg
        _MeshcatVisualizer = MeshcatVisualizer
    return _mg, _MeshcatVisualizer


def _lazy_import_weighted_moving_filter():
    from teleop.utils.weighted_moving_filter import WeightedMovingFilter
    return WeightedMovingFilter


# ── IK 结果 dataclass ─────────────────────────────────────────

class IKResult:
    """IK 求解结果 — 统一返回值。"""

    def __init__(self, arm_q: np.ndarray, arm_tau: np.ndarray,
                 head_q: Optional[np.ndarray] = None):
        self.arm_q = arm_q        # 双臂关节角 (14,)
        self.arm_tau = arm_tau    # 双臂力矩 (14,)
        self.head_q = head_q      # 头部关节角 (2,) 或 None

    def __repr__(self) -> str:
        return (f"IKResult(arm_q={self.arm_q.shape}, "
                f"head_q={self.head_q.shape if self.head_q is not None else None})")


class BaseArmIK(ABC):
    """所有机器人 IK 的公共基类。

    子类只需覆写：
      - _build_reduced_model()  — 加载 URDF、锁定关节、构建缩减模型
      - _setup_limits()         — 可选：覆盖 URDF 关节限位

    CasADi 符号模型构建、IPOPT 求解、平滑滤波、缓存等由基类完成。
    """

    def __init__(self, config: "RobotConfig", control_mode: "ControlMode",
                 visualization: str = "off", verbose: bool = False):
        from teleop.robot_control._base.robot_config import RobotConfig
        from teleop.robot_control._base.control_mode import ControlMode
        self.config: RobotConfig = config
        self.control_mode: ControlMode = control_mode
        self.visualization_mode = visualization
        self.Visualization = visualization != "off"
        self.verbose = verbose

        np.set_printoptions(precision=5, suppress=True, linewidth=200)

        # 缓存路径
        self.cache_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),  # subclasses may override
            self.config.cache_filename
        )

        # ── 子类填充 ──
        self.robot = None
        self.reduced_robot = None

        # ── 构建模型 ──
        self._load_or_build_model()
        self._build_joint_indices()
        self._setup_limits()
        self._build_casadi_model()
        self._build_optimization()
        self._init_solver_state()
        self._setup_visualization()

    # ─────────────── 子类必须覆写 ───────────────

    @abstractmethod
    def _build_reduced_model(self) -> tuple:
        """加载 URDF 并构建缩减模型。

        Returns:
            tuple: (robot: pin.RobotWrapper, reduced_robot: pin.RobotWrapper)
        """
        ...

    # ─────────────── 子类可选覆写 ───────────────

    def _setup_limits(self) -> None:
        """覆盖 URDF 关节限位。默认不做任何修改。"""
        pass

    def _setup_visualization(self) -> None:
        """设置 Meshcat 可视化。默认不做。"""
        pass

    def get_shoulder_positions(self, q=None) -> Optional[tuple]:
        """返回双肩位置 (l_xyz, r_xyz)。默认返回 None。"""
        return None

    # ─────────────── 公共方法 ──────────────────

    def solve_ik(self,
                 left_wrist_pose: np.ndarray,
                 right_wrist_pose: np.ndarray,
                 current_lr_arm_q: Optional[np.ndarray] = None,
                 current_lr_arm_dq: Optional[np.ndarray] = None,
                 head_target: Optional[np.ndarray] = None,
                 ros2_published_arm_q: Optional[np.ndarray] = None,
                 ) -> IKResult:
        """统一 IK 求解接口。

        子类可以直接使用此实现；如需自定义行为可覆写。
        """
        pin = _lazy_import_pin()

        # ── 初始化 ──
        init_q = np.array(self.init_data).reshape(-1)
        if init_q.size != self.reduced_robot.model.nq:
            init_q = np.zeros(self.reduced_robot.model.nq)

        if current_lr_arm_q is not None:
            current_lr_arm_q = np.array(current_lr_arm_q).reshape(-1)
            for i, idx in enumerate(self.arm_joint_indices):
                if i < current_lr_arm_q.size and idx < init_q.size:
                    init_q[idx] = current_lr_arm_q[i]

        self.opti.set_value(self.param_tf_l, self._DM(left_wrist_pose))
        self.opti.set_value(self.param_tf_r, self._DM(right_wrist_pose))

        # ── 头部处理 ──
        head_q_target = None
        if (self.control_mode.solve_head
                and self.head_rotation_error is not None
                and self.head_frame_id is not None):
            # ...existing head handling code (from TOPSTAR_H1_ArmIK.solve_ik)...
            pass  # subclass cán override full solve_ik if needed

        self.init_data = init_q
        self.opti.set_initial(self.var_q, self.init_data)
        self.opti.set_value(self.var_q_last, self.init_data)

        try:
            sol = self.opti.solve()
            sol_q_raw = np.array(sol.value(self.var_q)).reshape(-1)

            sol_q_arm_raw = sol_q_raw[self.arm_joint_indices]
            self.smooth_filter.add_data(sol_q_arm_raw)
            sol_q_arm = np.array(self.smooth_filter.filtered_data)

            # 头部提取
            if (self.control_mode.solve_head
                    and len(self.head_joint_map) > 0):
                head_q_list = []
                for _, _, idx, nq in self.head_joint_map:
                    head_q_list.append(sol_q_raw[idx: idx + nq])
                head_q_target = np.concatenate(head_q_list) if head_q_list else None
                if head_q_target is not None:
                    self.last_head_q = head_q_target.copy()

            v_full = np.zeros(self.reduced_robot.model.nv)
            sol_tauff_full = pin.rnea(
                self.reduced_robot.model, self.reduced_robot.data,
                sol_q_raw, v_full, np.zeros(self.reduced_robot.model.nv),
            )
            self.init_data = sol_q_raw.copy()

            return IKResult(
                arm_q=sol_q_arm,
                arm_tau=np.array(sol_tauff_full)[self.arm_joint_indices],
                head_q=head_q_target,
            )

        except Exception as e:
            if self.verbose:
                pin = _lazy_import_pin()
                print(f"[IK] solve failed: {e}, using debug solution")

            sol_q_raw = np.array(self.opti.debug.value(self.var_q)).reshape(-1)
            sol_q_arm_raw = sol_q_raw[self.arm_joint_indices]
            self.smooth_filter.add_data(sol_q_arm_raw)
            sol_q_arm = np.array(self.smooth_filter.filtered_data)

            if current_lr_arm_q is not None:
                return IKResult(
                    arm_q=np.array(current_lr_arm_q).reshape(-1)[:14],
                    arm_tau=np.zeros(14),
                    head_q=head_q_target if head_q_target is not None else None,
                )
            return IKResult(
                arm_q=sol_q_arm,
                arm_tau=np.zeros(len(sol_q_arm)),
                head_q=head_q_target,
            )

    def get_dual_arm_ee_positions(self, q_deg: np.ndarray):
        """正运动学 — 从关节角(度)计算末端位置。"""
        pin = _lazy_import_pin()
        q_rad = np.deg2rad(np.asarray(q_deg, dtype=float))
        q_full = np.zeros(self.reduced_robot.model.nq)
        for i, idx in enumerate(self.arm_joint_indices):
            q_full[idx] = q_rad[i]

        pin.forwardKinematics(self.reduced_robot.model, self.reduced_robot.data, q_full)
        pin.updateFramePlacements(self.reduced_robot.model, self.reduced_robot.data)
        left_pos = self.reduced_robot.data.oMf[self.L_hand_id].translation.copy()
        right_pos = self.reduced_robot.data.oMf[self.R_hand_id].translation.copy()
        return left_pos, right_pos

    def reset_smoothing(self, arm_q: np.ndarray) -> None:
        """Reset the command filter at a measured arm configuration."""
        value = np.asarray(arm_q, dtype=float).reshape(-1)
        expected = len(self.arm_joint_indices)
        if value.size < expected:
            raise ValueError(
                f"arm_q has {value.size} values, expected at least {expected}"
            )
        self.smooth_filter.reset(value[:expected])

    def get_dual_arm_ee_poses(self, q_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """正运动学 — 从关节角(度)计算完整末端位姿 (4×4)。

        返回:
            tuple: (left_ee_pose, right_ee_pose) 各为 (4,4) homogeneous 矩阵，robot 约定。
        """
        pin = _lazy_import_pin()
        q_rad = np.deg2rad(np.asarray(q_deg, dtype=float))
        q_full = np.zeros(self.reduced_robot.model.nq)
        for i, idx in enumerate(self.arm_joint_indices):
            q_full[idx] = q_rad[i]

        pin.forwardKinematics(self.reduced_robot.model, self.reduced_robot.data, q_full)
        pin.updateFramePlacements(self.reduced_robot.model, self.reduced_robot.data)
        left_pose = self.reduced_robot.data.oMf[self.L_hand_id].homogeneous.copy()
        right_pose = self.reduced_robot.data.oMf[self.R_hand_id].homogeneous.copy()
        return left_pose, right_pose

    def save_cache(self) -> None:
        """保存 Pinocchio 模型缓存。"""
        import pickle
        cache_data = {
            "cache_version": 2,
            "limit_mode": self.config.limit_mode,
            "robot_model": self.robot.model,
            "reduced_model": self.reduced_robot.model,
        }
        tmp_path = self.cache_path + ".tmp"
        with open(tmp_path, "wb") as f:
            pickle.dump(cache_data, f)
        os.replace(tmp_path, self.cache_path)

    def load_cache(self, expected_limit_mode: Optional[str] = None):
        """加载 Pinocchio 模型缓存。"""
        import pickle
        pin = _lazy_import_pin()
        try:
            with open(self.cache_path, "rb") as f:
                cache = pickle.load(f)
            if not isinstance(cache, dict):
                raise ValueError(f"unexpected cache payload type: {type(cache)!r}")
            if "robot_model" not in cache or "reduced_model" not in cache:
                raise KeyError("missing robot_model or reduced_model")
            if expected_limit_mode is not None:
                if cache.get("limit_mode") != expected_limit_mode:
                    raise ValueError(
                        f"cache limit mode mismatch: expected {expected_limit_mode!r}, "
                        f"got {cache.get('limit_mode')!r}"
                    )
            robot = pin.RobotWrapper()
            robot.model = cache["robot_model"]
            robot.data = robot.model.createData()
            reduced_robot = pin.RobotWrapper()
            reduced_robot.model = cache["reduced_model"]
            reduced_robot.data = reduced_robot.model.createData()
            return robot, reduced_robot
        except Exception as exc:
            warnings.warn(f"Failed to load cache at {self.cache_path}: {exc}. Rebuilding.", RuntimeWarning)
            try:
                os.remove(self.cache_path)
            except OSError:
                pass
            return None

    # ─────────────── 内部方法 ──────────────────

    def _load_or_build_model(self) -> None:
        """先尝试缓存加载，失败则调用子类的 _build_reduced_model()。"""
        if os.path.exists(self.cache_path) and not self.Visualization:
            cached = self.load_cache(expected_limit_mode=self.config.limit_mode)
            if cached is not None:
                self.robot, self.reduced_robot = cached
                return
        self.robot, self.reduced_robot = self._build_reduced_model()
        if not self.Visualization:
            self.save_cache()

    def _build_joint_indices(self) -> None:
        """根据 config 中的关节名构建索引。"""
        self.arm_joint_indices = []
        self.l_arm_joint_indices = []
        self.r_arm_joint_indices = []

        model = self.reduced_robot.model
        for name in self.config.left_arm_joint_names:
            if not model.existJointName(name):
                continue
            jid = model.getJointId(name)
            j = model.joints[jid]
            for k in range(j.nq):
                self.arm_joint_indices.append(j.idx_q + k)
                self.l_arm_joint_indices.append(j.idx_q + k)

        for name in self.config.right_arm_joint_names:
            if not model.existJointName(name):
                continue
            jid = model.getJointId(name)
            j = model.joints[jid]
            for k in range(j.nq):
                self.arm_joint_indices.append(j.idx_q + k)
                self.r_arm_joint_indices.append(j.idx_q + k)

        # 头部关节映射
        self.head_joint_map: list[tuple] = []
        self.head_q_indices: list[int] = []
        self.last_head_q: Optional[np.ndarray] = None

        if (self.control_mode.solve_head
                and self.config.head_joint_names is not None):
            total_head_dof = 0
            for name in self.config.head_joint_names:
                if model.existJointName(name):
                    jid = model.getJointId(name)
                    j = model.joints[jid]
                    idx = getattr(j, 'idx_q', 0)
                    nq = getattr(j, 'nq', 1)
                    self.head_joint_map.append((name, jid, idx, nq))
                    self.head_q_indices.append(idx)
                    total_head_dof += nq
            self.last_head_q = np.zeros(total_head_dof) if total_head_dof > 0 else np.zeros(0)

        # EE frame id
        self.L_hand_id = model.getFrameId(self.config.left_ee_frame_name)
        self.R_hand_id = model.getFrameId(self.config.right_ee_frame_name)

    def _build_casadi_model(self) -> None:
        """构建 CasADi 符号模型（所有机器人通用）。"""
        cpin = _lazy_import_cpin()
        casadi = _lazy_import_casadi()

        self.cmodel = cpin.Model(self.reduced_robot.model)
        self.cdata = self.cmodel.createData()

        self.cq = casadi.SX.sym("q", self.reduced_robot.model.nq, 1)
        self.cTf_l = casadi.SX.sym("tf_l", 4, 4)
        self.cTf_r = casadi.SX.sym("tf_r", 4, 4)

        cpin.framesForwardKinematics(self.cmodel, self.cdata, self.cq)

        self.translational_error = casadi.Function(
            "translational_error",
            [self.cq, self.cTf_l, self.cTf_r],
            [casadi.vertcat(
                self.cdata.oMf[self.L_hand_id].translation - self.cTf_l[:3, 3],
                self.cdata.oMf[self.R_hand_id].translation - self.cTf_r[:3, 3],
            )],
        )
        self.rotational_error = casadi.Function(
            "rotational_error",
            [self.cq, self.cTf_l, self.cTf_r],
            [casadi.vertcat(
                cpin.log3(self.cdata.oMf[self.L_hand_id].rotation @ self.cTf_l[:3, :3].T),
                cpin.log3(self.cdata.oMf[self.R_hand_id].rotation @ self.cTf_r[:3, :3].T),
            )],
        )

        # Head frame（可选）
        self.head_frame_id = None
        self.head_rotation_error = None
        if (self.control_mode.solve_head
                and self.config.head_frame_name is not None
                and self.reduced_robot.model.existFrame(self.config.head_frame_name)):
            self.head_frame_id = self.reduced_robot.model.getFrameId(
                self.config.head_frame_name
            )
            self.cTf_head = casadi.SX.sym("tf_head", 4, 4)
            self.head_rotation_error = casadi.Function(
                "head_rotation_error",
                [self.cq, self.cTf_head],
                [cpin.log3(
                    self.cdata.oMf[self.head_frame_id].rotation @ self.cTf_head[:3, :3].T
                )],
            )

    def _build_optimization(self) -> None:
        """构建 IPOPT 优化问题（所有机器人通用）。"""
        casadi = _lazy_import_casadi()

        self.opti = casadi.Opti()
        self.var_q = self.opti.variable(self.reduced_robot.model.nq)
        self.var_q_last = self.opti.parameter(self.reduced_robot.model.nq)
        self.param_tf_l = self.opti.parameter(4, 4)
        self.param_tf_r = self.opti.parameter(4, 4)

        translational_cost = casadi.sumsqr(
            self.translational_error(self.var_q, self.param_tf_l, self.param_tf_r)
        )
        rotation_cost = casadi.sumsqr(
            self.rotational_error(self.var_q, self.param_tf_l, self.param_tf_r)
        )
        regularization_cost = casadi.sumsqr(self.var_q)
        smooth_cost = casadi.sumsqr(self.var_q - self.var_q_last)

        # The normal safe-margin solver retains the historical weak pull
        # toward zero.  Hard-limit startup is used for measured HOME poses
        # inside the mechanical envelope but outside that margin.  Pulling
        # those poses toward zero would create motion on the very first IK
        # frame even when both wrist targets are unchanged, so hard mode uses
        # only the measured-state smoothness term as joint regularization.
        zero_pose_regularization_weight = (
            0.0 if self.config.limit_mode == "hard" else 0.02
        )

        total_cost = (
            50.0 * translational_cost
            + 1.0 * rotation_cost
            + zero_pose_regularization_weight * regularization_cost
            + 0.1 * smooth_cost
        )

        # 头部代价
        self.param_tf_head = None
        if self.head_rotation_error is not None:
            self.param_tf_head = self.opti.parameter(4, 4)
            head_pose_cost = casadi.sumsqr(
                self.head_rotation_error(self.var_q, self.param_tf_head)
            )
            total_cost = total_cost + 4.0 * head_pose_cost

        self.opti.subject_to(self.opti.bounded(
            self.reduced_robot.model.lowerPositionLimit,
            self.var_q,
            self.reduced_robot.model.upperPositionLimit,
        ))
        self.opti.minimize(total_cost)

        opts = {
            "expand": True,
            "detect_simple_bounds": True,
            "calc_lam_p": False,
            "print_time": False,
            "ipopt.sb": "yes",
            "ipopt.print_level": 0,
            "ipopt.max_iter": 30,
            "ipopt.tol": 1e-4,
            "ipopt.acceptable_tol": 5e-4,
            "ipopt.acceptable_iter": 5,
            "ipopt.warm_start_init_point": "yes",
            "ipopt.derivative_test": "none",
            "ipopt.jacobian_approximation": "exact",
        }
        self.opti.solver("ipopt", opts)

    def _init_solver_state(self) -> None:
        """初始化求解器状态和滤波器。"""
        WeightedMovingFilter = _lazy_import_weighted_moving_filter()
        arm_dof = len(self.arm_joint_indices)

        self.init_data = np.zeros(self.reduced_robot.model.nq)
        self.opti.set_initial(self.var_q, self.init_data)
        self.opti.set_value(self.var_q_last, self.init_data)

        self.smooth_filter = WeightedMovingFilter(
            np.array([0.4, 0.3, 0.2, 0.1]), arm_dof
        )

    @staticmethod
    def _DM(matrix: np.ndarray):
        """将 numpy 矩阵转为 CasADi DM。"""
        casadi = _lazy_import_casadi()
        return casadi.DM(matrix)
