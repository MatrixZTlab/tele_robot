h2机器人仿真环境
ros2 run topstar_ros2_example mujoco_ros2_bridge
仿真镜像
/media/ai/d9787eb9-5947-4134-be08-d9b5ed71bdde/tele_robot/complete_test/topstar_ros2_redeploy_20260529/topstar_mujoco/simulate/build/topstar_mujoco -n lo --lowstate
h2机器人实机小脑控制程序
cd topstar_h2
python3 python/motor_gui.py -f ec_rt.conf
