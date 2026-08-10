#!/usr/bin/env bash
set -eo pipefail

source /home/ai/yes/etc/profile.d/conda.sh
conda activate tv

source /home/ai/lium/topstar_ros2_redeploy_20260626/setup.sh
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

cd /home/ai/lium/tele_robot

teleop_task_name="${1:-topstar_h1_box_new_003}"

exec python -m teleop.teleop_h1_vr_mujoco_live \
  --live \
  --allow-hard-limit-start \
  --frequency 20 \
  --controller-mapping mirrored \
  --arm-scale 1.2 \
  --ema-alpha 0.8 \
  --position-deadband-m 0.003 \
  --rotation-deadband-deg 1.0 \
  --max-joint-speed 0.25 \
  --max-solver-jump 0.20 \
  --max-tracking-error 0.20 \
  --max-arm-speed-at-arm 0.20 \
  --state-timeout 0.25 \
  --pico-timeout 0.25 \
  --max-ik-position-error 0.06 \
  --max-ik-rotation-error-deg 5.0 \
  --record \
  --no-rerun \
  --img-server-ip 192.168.31.3 \
  --task-name "${teleop_task_name}"
