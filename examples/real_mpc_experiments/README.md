# 真机 MPC 实验

这个目录用于存放真机 MPC 调试脚本。

推荐调试顺序：

1. `EX29_vbot_dds_lowstate_monitor.py`：只读 DDS lowstate 监控。订阅
   `rt/lowstate`，把电机反馈映射到 model 坐标，不发布 `rt/lowcmd`。
2. `EX30_vbot_real_mpc_monitor.py`：只监控 MPC bridge。读取真机电机反馈，
   映射到模型状态，求解 MPC，只打印 `tau_model` / `tau_motor`，不使能电机，
   不发送非零控制。
3. `EX31_vbot_real_mpc_torque_test.py`：悬空低力矩测试。强制 `yaw=0`、
   `wz=0`，机身速度命令为 0，对 `tau_motor` 做限幅，只发送 torque-only
   串口帧。
4. 再做站立支撑测试。
5. 前面阶段稳定后，再做原地踏步或慢走。

EX29 只读监控示例：

```bash
python3 examples/real_mpc_experiments/EX29_vbot_dds_lowstate_monitor.py \
  --network lo \
  --duration 5 \
  --prone-calibrate-on-start
```

EX30 只监控 MPC 示例：

```bash
python3 examples/real_mpc_experiments/EX30_vbot_real_mpc_monitor.py \
  --duration 10 \
  --prone-calibrate-on-start
```

EX31 悬空低力矩示例：

```bash
python3 examples/real_mpc_experiments/EX31_vbot_real_mpc_torque_test.py \
  --affine configs/vbot_real_joint_affine.yaml \
  --duration 3 \
  --mpc-hz 10 \
  --cmd-hz 100 \
  --tau-limit 0.5 \
  --prone-calibrate-on-start \
  --robot-is-suspended \
  --send-enable \
  --disable-on-exit \
  --i-accept-risk
```

预期现象：腿部只有小幅、有界的力矩输出。如果任何腿突然抽动、旋转、
振荡，或运动方向和已验证关节方向相反，立即停止。

## EX33B 真机调试命令

更完整的 EX33B 流程见 `EX33B_TEST_FLOW.md`。这里放现场调试最常用的命令。

### 1. PC 同步代码

在 PC 上同步最新真机实验脚本到 OrangePi。每次改过
`EX33B_vbot_dds_stand_mpc_overlay.py` 或
`EX34_vbot_dds_real_mpc_state_estimator.py` 后都要先同步，再去 OrangePi 上跑。

```bash
cd /home/lushilin/vbot_mpc_ws

rsync -avR \
  go2-convex-mpc/examples/real_mpc_experiments/EX33B_vbot_dds_stand_mpc_overlay.py \
  go2-convex-mpc/examples/real_mpc_experiments/EX34_vbot_dds_real_mpc_state_estimator.py \
  orangepi@192.168.3.93:~/
```

如果 README 或测试说明也改了，可以一起同步：

```bash
cd /home/lushilin/vbot_mpc_ws

rsync -avR \
  go2-convex-mpc/examples/real_mpc_experiments/README.md \
  go2-convex-mpc/examples/real_mpc_experiments/EX33B_TEST_FLOW.md \
  orangepi@192.168.3.93:~/
```

### 2. OrangePi 启动环境

真机测试建议开两个 OrangePi 终端：

- 终端 1：启动 DDS/serial gateway。
- 终端 2：运行 EX33B Python 测试脚本。

两个终端都先登录 OrangePi：

```bash
ssh orangepi@192.168.3.93
```

终端 2 需要进入 Python 环境。如果提示符里已经有
`(vbot-real-mpc)`，说明环境已经激活，可以跳过 `conda activate`。

```bash
cd ~/go2-convex-mpc
unset PYTHONPATH
source ~/miniforge3/etc/profile.d/conda.sh
conda env list
conda activate vbot-real-mpc

export CYCLONEDDS_HOME=/usr/local
export CMAKE_PREFIX_PATH=/usr/local
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
export PYTHONNOUSERSITE=1

python3 - <<'PY'
import convex_mpc
print("python env ok")
PY
```

如果 `conda activate vbot-real-mpc` 找不到环境，先用 `conda env list`
确认实际环境名，再替换上面的环境名。

### 3. 启动 gateway

在 OrangePi 终端 1 启动 DDS/serial gateway：

```bash
cd ~/fatuDog/serial_dds_gateway

./build/dds_to_serial_gateway \
  --serial-port-a /dev/myttyCAN0 \
  --serial-port-b /dev/myttyCAN1 \
  --baudrate 2000000 \
  --network lo \
  --tick-hz 100 \
  --imu-port /dev/myttyIMU \
  --imu-baudrate 921600 \
  --imu-gyro-deadzone 0.005
```

gateway 正常后，终端里应该持续打印 IMU / lowstate / CAN 相关信息。
如果 EX33B 提示收不到 `rt/lowstate`，先检查这个终端是否还在运行。

### 4. 运行 EX33B MPC overlay

在 OrangePi 终端 2 运行 EX33B MPC overlay：

```bash
cd ~/go2-convex-mpc
unset PYTHONPATH
source ~/miniforge3/etc/profile.d/conda.sh
conda activate vbot-real-mpc

export CYCLONEDDS_HOME=/usr/local
export CMAKE_PREFIX_PATH=/usr/local
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
export PYTHONNOUSERSITE=1

python3 examples/real_mpc_experiments/EX33B_vbot_dds_stand_mpc_overlay.py \
  --network lo \
  --target-pose stand \
  --prone-calibrate-on-start \
  --prone-pose down \
  --startup-ramp-seconds 16 \
  --prehold-seconds 2 \
  --duration 10 \
  --cmd-hz 100 \
  --mpc-hz 5 \
  --kp 50 \
  --kd 3 \
  --final-kp 30 \
  --final-kd 2.5 \
  --handover-seconds 4 \
  --tau-limit 0.05 \
  --final-tau-limit 1.5 \
  --adaptive-tau-limit \
  --tau-limit-rate 0.3 \
  --adaptive-tilt-soft 0.025 \
  --adaptive-tilt-hard 0.070 \
  --abort-on-large-error \
  --abort-qerr 0.50 \
  --abort-tilt 0.08 \
  --use-imu-base-state \
  --imu-rp-zero-on-start \
  --imu-rp-zero-seconds 0.5 \
  --imu-gyro-deadband 0.005 \
  --fix-gateway-gyro-order \
  --mpc-r 3e-3 \
  --tau-limit-mode scale \
  --ramp-seconds 3 \
  --return-pose-on-exit down \
  --return-ramp-seconds 5 \
  --disable-on-exit \
  --allow-large-gains \
  --allow-large-tau-limit \
  --allow-large-gains \
  --robot-standing-supported \
  --i-accept-risk
```

运行 EX33B 单腿抬腿调试。传入 `--lift-leg FL` 后，默认 MPC overlay 内部
流程是：2 秒四足站立、2 秒抬腿、2 秒保持、2 秒放回。

```bash
cd ~/go2-convex-mpc
unset PYTHONPATH
source ~/miniforge3/etc/profile.d/conda.sh
conda activate vbot-real-mpc

export CYCLONEDDS_HOME=/usr/local
export CMAKE_PREFIX_PATH=/usr/local
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
export PYTHONNOUSERSITE=1

python3 examples/real_mpc_experiments/EX33B_vbot_dds_stand_mpc_overlay.py \
  --network lo \
  --target-pose stand \
  --prone-calibrate-on-start \
  --prone-pose down \
  --startup-ramp-seconds 16 \
  --prehold-seconds 2 \
  --duration 10 \
  --cmd-hz 100 \
  --mpc-hz 5 \
  --kp 50 \
  --kd 3 \
  --final-kp 30 \
  --final-kd 2.5 \
  --handover-seconds 4 \
  --tau-limit 0.05 \
  --final-tau-limit 1.5 \
  --adaptive-tau-limit \
  --tau-limit-rate 0.3 \
  --adaptive-tilt-soft 0.025 \
  --adaptive-tilt-hard 0.070 \
  --abort-on-large-error \
  --abort-qerr 0.50 \
  --abort-tilt 0.08 \
  --use-imu-base-state \
  --imu-rp-zero-on-start \
  --imu-rp-zero-seconds 0.5 \
  --imu-gyro-deadband 0.005 \
  --fix-gateway-gyro-order \
  --mpc-r 3e-3 \
  --tau-limit-mode scale \
  --lift-leg FL \
  --lift-thigh-offset 0.08 \
  --lift-calf-offset -0.16 \
  --allow-single-leg-lift \
  --ramp-seconds 3 \
  --return-pose-on-exit down \
  --return-ramp-seconds 5 \
  --disable-on-exit \
  --allow-large-gains \
  --allow-large-tau-limit \
  --allow-large-gains \
  --robot-standing-supported \
  --i-accept-risk
```

### 4B. 运行 EX34 real-state MPC

EX34 是“仿真 MPC 控制结构”的真机入口：先用 IMU roll/pitch/gyro、
关节反馈和支撑脚运动学更新 Pinocchio floating-base state，再运行
`ComTraj + CentroidalMPC + LegController`。第一轮只跑 all-stance，不打开
trot/swing。

```bash
cd ~/go2-convex-mpc
unset PYTHONPATH
source ~/miniforge3/etc/profile.d/conda.sh
conda activate vbot-real-mpc

export CYCLONEDDS_HOME=/usr/local
export CMAKE_PREFIX_PATH=/usr/local
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
export PYTHONNOUSERSITE=1

python3 examples/real_mpc_experiments/EX34_vbot_dds_real_mpc_state_estimator.py \
  --network lo \
  --target-pose stand \
  --prone-calibrate-on-start \
  --prone-pose down \
  --startup-ramp-seconds 12 \
  --prehold-seconds 2 \
  --duration 6 \
  --cmd-hz 100 \
  --state-hz 50 \
  --mpc-hz 5 \
  --kp 50 \
  --kd 3 \
  --final-kp 50 \
  --final-kd 3 \
  --handover-seconds 0 \
  --tau-limit 0.03 \
  --final-tau-limit 0.6 \
  --adaptive-tau-limit \
  --tau-limit-rate 0.2 \
  --adaptive-tilt-soft 0.025 \
  --adaptive-tilt-hard 0.070 \
  --abort-on-large-error \
  --abort-qerr 0.50 \
  --abort-tilt 0.06 \
  --use-imu-base-state \
  --imu-rp-zero-on-start \
  --imu-rp-zero-seconds 0.5 \
  --imu-gyro-deadband 0.005 \
  --fix-gateway-gyro-order \
  --base-height-mode stance-feet \
  --base-vel-mode stance-feet \
  --gait all-stance \
  --mpc-r 3e-3 \
  --tau-limit-mode scale \
  --ramp-seconds 2 \
  --return-pose-on-exit down \
  --return-ramp-seconds 5 \
  --disable-on-exit \
  --allow-large-gains \
  --robot-standing-supported \
  --i-accept-risk
```

EX34 当前把 100Hz lowcmd 发布放在主循环最高优先级，状态估计默认
50Hz、MPC 默认 5Hz。跑完优先看 summary 的 lowcmd 计数和
publish/MPC/state 耗时，确认瓶颈在 Python 侧还是 DDS/gateway 侧。

### 5. 拉取日志

把 CSV 日志和 summary 拉回 PC：

```bash
cd /home/lushilin/vbot_mpc_ws
mkdir -p debug_logs/orangepi_real_mpc

rsync -av \
  orangepi@192.168.3.93:/home/orangepi/go2-convex-mpc/logs/real_mpc/<log>.csv \
  debug_logs/orangepi_real_mpc/

rsync -av \
  orangepi@192.168.3.93:/home/orangepi/go2-convex-mpc/logs/real_mpc/<log>_summary.txt \
  debug_logs/orangepi_real_mpc/
```

把 `<log>` 替换成 EX33B 打印出的日志文件名主干，例如
`ex33b_stand_mpc_overlay_20260616_095445`。

如果 MPC overlay 提前退出，优先看 summary：

```bash
cat debug_logs/orangepi_real_mpc/<log>_summary.txt
```

重点字段：

```text
exit_reason
overlay_elapsed_s
overlay_lowcmd_count
overlay_lowcmd_interval_mean_s
overlay_lowcmd_interval_max_s
```

单腿抬腿成功进入流程时，CSV 应该能看到：

```text
contact_mask: 1111 -> 0111 -> 1111
lift_alpha: 0 -> 1 -> 0
```
