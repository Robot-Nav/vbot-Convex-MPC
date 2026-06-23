# EX33B 真机站立 MPC Overlay 测试流程

本文记录 `EX33B_vbot_dds_stand_mpc_overlay.py` 当前推荐的真机测试流程。

当前控制链路：

```text
Python EX33B
  订阅 rt/lowstate
  读取 model-space 关节反馈 + IMU roll/pitch/gyro
  使用 PD 保持 stand 姿态
  计算 MPC torque overlay
  使用 adaptive tau limit + torque-vector scale 做安全限幅
  发布 rt/lowcmd

C++ dds_to_serial_gateway
  独占 /dev/myttyCAN0 /dev/myttyCAN1 /dev/myttyIMU
  将 model-space 关节命令映射到 motor-space
  发布 rt/lowstate
```

安全前提：

```text
机器人必须有人扶住或有可靠支撑。
先确认 emergency stop / 断电方式可用。
每次只改一个参数，避免同时引入多个变量。
```

## 1. 文件传输命令

在 PC 上运行：

```bash
cd /home/lushilin/vbot_mpc_ws

rsync -avR \
  go2-convex-mpc/examples/real_mpc_experiments/EX33B_vbot_dds_stand_mpc_overlay.py \
  go2-convex-mpc/examples/real_mpc_experiments/EX34_vbot_dds_real_mpc_state_estimator.py \
  orangepi@192.168.3.93:~/
```

可选：在 PC 上记录 md5，用来和 OrangePi 上的文件对比。

```bash
md5sum go2-convex-mpc/examples/real_mpc_experiments/EX33B_vbot_dds_stand_mpc_overlay.py
```

## 2. Gateway 启动命令

在 OrangePi 终端 1 运行：

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

启动后应看到：

```text
IMU serial enabled on /dev/myttyIMU @921600
imu_frames 持续增加
imu_errors=0
EX33B 开始发布后 rx_frames 持续增加
```

注意：当前 STM32 IMU 串口帧顺序是：

```text
yaw, pitch, roll, gx, gy, gz
```

部分 gateway 版本会错误解析成：

```text
yaw, pitch, roll, gz, gy, gx
```

所以当前 EX33B 推荐测试命令里保留 `--fix-gateway-gyro-order` 作为临时修正。

## 3. EX33B 当前推荐测试命令

在 OrangePi 终端 2 运行：

```bash
cd ~/go2-convex-mpc

python3 examples/real_mpc_experiments/EX33B_vbot_dds_stand_mpc_overlay.py \
  --network lo \
  --target-pose stand \
  --prone-calibrate-on-start \
  --prone-pose down \
  --startup-ramp-seconds 12 \
  --prehold-seconds 2 \
  --duration 10 \
  --cmd-hz 100 \
  --mpc-hz 5 \
  --kp 50 \
  --kd 3 \
  --final-kp 30 \
  --final-kd 2.5 \
  --tau-limit 0.10 \
  --final-tau-limit 2.5 \
  --adaptive-tau-limit \
  --tau-limit-rate 0.5 \
  --adaptive-tilt-soft 0.025 \
  --adaptive-tilt-hard 0.070 \
  --abort-on-large-error \
  --abort-tilt 0.08 \
  --use-imu-base-state \
  --imu-rp-zero-on-start \
  --imu-rp-zero-seconds 0.5 \
  --imu-gyro-deadband 0.005 \
  --fix-gateway-gyro-order \
  --mpc-r 1e-3 \
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

EX33B 启动时应看到：

```text
Base state: IMU roll/pitch + gyro gx/gy, yaw=0 and gz=0
IMU roll/pitch zero-on-start enabled
IMU roll/pitch zero-on-start: roll0=... pitch0=...
IMU gyro order workaround: DDS [gz,gy,gx] -> controller [gx,gy,gz]
MPC input cost R diagonal: 1.000e-03
Torque limit mode: scale
```

如果没有看到 `IMU gyro order workaround`，说明没有启用 `--fix-gateway-gyro-order`。

## 4. 日志拉取命令

EX33B 结束后会打印类似：

```text
log_csv_saved: /home/orangepi/go2-convex-mpc/logs/real_mpc/<log>.csv
```

回到 PC 运行：

```bash
cd /home/lushilin/vbot_mpc_ws
mkdir -p debug_logs/orangepi_real_mpc

rsync -av \
  orangepi@192.168.3.93:/home/orangepi/go2-convex-mpc/logs/real_mpc/<log>.csv \
  debug_logs/orangepi_real_mpc/
```

把 `<log>.csv` 替换成 EX33B 打印出的实际文件名。

## 5. 日志里要看哪些指标

当前 CSV 默认只保存 `mpc_overlay` 行，不再保存 `stand_ramp` 和 `return_ramp`。PD 起立和退出回落仍会正常执行，只是不混进日志。

如果 EX33B 打印：

```text
log_csv_saved: none (MPC overlay did not start)
```

说明程序在进入 MPC overlay 前就退出了。

每次 MPC overlay 结束还会打印并保存：

```text
summary_saved: ..._summary.txt
```

这个 summary 里会记录 `exit_reason`、`overlay_elapsed_s`、`overlay_lowcmd_count` 和 lowcmd 发送间隔。若 overlay 又提前结束，优先看这个文件。

优先看这些 CSV 列：

```text
elapsed_s
phase
joint
q_model
q_target
q_err
tau_raw
tau_clip
tau_cmd
tau_est
alpha
clipped_count
mpc_solve_ms
handover_alpha
kp_cmd
kd_cmd
tau_limit_cmd
base_roll
base_pitch
base_gx
base_gy
base_gz
```

好的现象：

```text
base_roll 保持较小，理想范围约 +/-0.03 rad 内
base_pitch 保持较小
max abs(q_err) 最好低于约 0.10 rad
tau_limit_cmd 平滑上升，没有突跳
tau_cmd 不应该在大多数关节上长时间顶到限幅
clipped_count 大多数时刻应较低，不应长期接近 12
mpc_solve_ms 稳定，不应频繁超过 MPC 周期
return_ramp 能平滑回到 down 姿态
```

当前较好的参考结果：

```text
sat_frac 从约 46.7% 降到约 5.8%
max q_err 从约 0.089 rad 降到约 0.075 rad
base_roll final 从约 0.027 rad 降到约 0.019 rad
```

仍需重点关注：

```text
机器人运动时 base_gx/base_gy 不应该始终精确等于 0。
base_gz 当前由 EX33B 强制置 0，这是预期行为。
```

## 6. 侧偏怎么处理

先保持当前安全 MPC 设置：

```bash
--mpc-r 1e-3
--tau-limit-mode scale
--final-tau-limit 2.5
```

如果机器人持续向同一侧偏，优先检查：

```text
1. base_roll 是否有明显非零偏置。
2. q_err 是否在某一侧腿或某几个关节长期偏大。
3. tau_cmd 是否某一侧长期更大或更容易触发限幅。
4. base_gx/base_gy 是否正常，不应全程为 0。
```

如果怀疑 roll 符号反了，单独测试：

```bash
--imu-roll-sign -1
```

如果改 roll sign 后侧偏明显反向或改善，说明 roll 坐标约定需要继续确认。测试时只改这一项，其他参数保持不变。gateway parser 未修好前继续保留 `--fix-gateway-gyro-order`。

## 7. 振荡或突然跳动怎么处理

出现明显振荡、突然顶腿、机身快速摆动时，先降低 MPC 权限：

```bash
--final-tau-limit 1.5
--tau-limit-rate 0.3
--mpc-r 3e-3
```

继续保留退出和保护参数：

```bash
--return-pose-on-exit down
--return-ramp-seconds 5
--disable-on-exit
--abort-on-large-error
--abort-tilt 0.08
```

日志判断重点：

```text
tau_limit_cmd 是否上升太快
clipped_count 是否长期偏高
tau_cmd 是否在某些关节正负来回打满
base_roll/base_pitch 是否接近 abort-tilt
q_err 是否在振荡开始前已经变大
```

如果降低 MPC 权限后稳定，说明当前 MPC overlay 太强；下一步再逐步增大 `--final-tau-limit` 或减小 `--mpc-r`，不要同时调两项。

## 8. 角速度为 0 时怎么处理

日志里如果 `base_gx` 和 `base_gy` 全程都是 `0.000000000`，按顺序检查：

```text
1. gateway 是否带了 --imu-gyro-deadzone 0.005。
2. EX33B 是否带了 --imu-gyro-deadband 0.005。
3. 手动轻微晃动机身时，gateway 的 IMU 帧是否还在增加。
4. 是否启用了 --use-imu-base-state。
5. 是否启用了当前推荐的 --fix-gateway-gyro-order。
```

如果 `base_gx/base_gy` 仍然为 0：

```text
gateway 可能已经在发布前把 gyro 清零；
gateway 也可能仍在错误解析 gx/gz 顺序；
需要优先修 gateway 的 IMU parser，再继续放大 MPC 权限。
```

如果只有小角速度被清零，可以临时降低死区：

```bash
# gateway
--imu-gyro-deadzone 0.002

# EX33B
--imu-gyro-deadband 0.002
```

但不要直接设成很大的 deadband；过大的 deadband 会让小幅 roll/pitch rate 被吃掉，MPC 看到的机身角速度会失真。

## 9. 关键参数作用说明

`--tau-limit-mode scale`

```text
当 MPC 原始 torque 向量超过 tau limit 时，按同一个比例缩放整条 torque 向量。
这样可以保留 MPC 算出的关节 torque 相对比例和方向。
当前推荐使用 scale。
```

`--tau-limit-mode clip`

```text
旧方式：每个关节独立硬裁剪到 [-tau_limit, +tau_limit]。
缺点是会破坏 MPC torque 向量比例，可能带来左右侧 bias 或非预期扭矩方向。
```

`--mpc-r`

```text
MPC 输入力/力矩代价 R 的对角线值。
数值越大，MPC 越不愿意用大力，overlay 越温和。
数值越小，MPC 越激进，更容易修正姿态，也更容易引入振荡或限幅。
当前推荐 1e-3；振荡时可先试 3e-3。
```

`--fix-gateway-gyro-order`

```text
临时修正部分 gateway 对 IMU gyro 顺序的解析错误。
启用后 EX33B 会把 DDS 中的 [gz, gy, gx] 当作 controller 需要的 [gx, gy, gz]。
当 gateway parser 已经修成正确的 gx,gy,gz 顺序后，应去掉这个参数。
```

## 10. 当前边界

```text
EX33B 仍然是 stand PD + MPC torque overlay。
它不是 walking MPC。
走路还需要 gait/contact switching 和 swing leg trajectory，不是只把这个站立 overlay 放大。
```

## 11. 阶段 2：增强 MPC 接管

目标：在仍然四足站立的情况下，让 MPC 承担更多姿态/支撑修正，PD 逐步退到保底。

先从 2A 开始：

```bash
python3 examples/real_mpc_experiments/EX33B_vbot_dds_stand_mpc_overlay.py \
  --network lo \
  --target-pose stand \
  --prone-calibrate-on-start \
  --prone-pose down \
  --startup-ramp-seconds 12 \
  --prehold-seconds 2 \
  --duration 10 \
  --cmd-hz 100 \
  --mpc-hz 3 \
  --kp 50 \
  --kd 3 \
  --final-kp 25 \
  --final-kd 2.0 \
  --tau-limit 0.10 \
  --final-tau-limit 3.0 \
  --adaptive-tau-limit \
  --tau-limit-rate 0.5 \
  --adaptive-tilt-soft 0.025 \
  --adaptive-tilt-hard 0.070 \
  --abort-on-large-error \
  --abort-tilt 0.08 \
  --use-imu-base-state \
  --imu-gyro-deadband 0.005 \
  --fix-gateway-gyro-order \
  --mpc-r 1e-3 \
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

2A 稳定后，再测 2B：

```bash
--mpc-hz 5
--final-kp 20
--final-kd 1.5
--final-tau-limit 3.5
--tau-limit-rate 0.4
--mpc-r 8e-4
```

2B 稳定后才测 2C：

```bash
--final-kp 15
--final-kd 1.2
--final-tau-limit 4.0
--tau-limit-rate 0.3
--mpc-r 7e-4
```

进入下一档的条件：

```text
base_roll/base_pitch 不持续变大
max abs(q_err) < 0.10 rad
clipped_count 不长期 12/12
tau_cmd 不正负来回打满
base_gx/base_gy 不是全 0
现场没有侧偏加剧或振荡
```

## 12. 阶段 3A：站立抗扰动

先不抬腿。使用阶段 2B 参数，把 `--duration` 增到 15 秒：

```bash
--duration 15
--allow-long-duration
```

测试方法：

```text
1. 四脚都接触地面，保持 all-stance。
2. 3 秒后轻推 roll 方向。
3. 6 秒后轻推 pitch 方向。
4. 观察机身是否回正。
```

成功标准：

```text
扰动后 base_roll/base_pitch 能回到小范围
不触发 abort-tilt
q_err 不突然超过 0.10~0.12 rad
clipped_count 可以短时升高，但不能长期满限幅
```

## 13. 阶段 3B：单腿抬起/三足支撑

EX33B 已增加默认关闭的单腿抬腿实验模式：

```text
--lift-leg FL/FR/RL/RR
--lift-hip-offset
--lift-thigh-offset
--lift-calf-offset
--allow-single-leg-lift
```

只要传 `--lift-leg`，默认 MPC overlay 内部流程就是：

```text
MPC 0.0s ~ 2.0s   正常四足站立
MPC 2.0s ~ 4.0s   开始抬腿，lift_alpha 0 -> 1
MPC 4.0s ~ 6.0s   保持抬腿
MPC 6.0s ~ 8.0s   放回，lift_alpha 1 -> 0
```

如需特殊实验，再手动覆盖 `--lift-start-s`、`--lift-ramp-s`、`--lift-hold-s`。

控制方式：

```text
指定腿的 q_target 从 stand 平滑插值到 lift pose；
MPC contact_mask 中指定腿置 0；
指定腿的 MPC torque 强制置 0，只用 joint PD 保持抬腿姿态；
其余三条腿继续使用 MPC torque overlay 做站立支撑。
```

建议先抬前腿，从 FL 或 FR 选一个。第一次推荐保守命令：

```bash
python3 examples/real_mpc_experiments/EX33B_vbot_dds_stand_mpc_overlay.py \
  --network lo \
  --target-pose stand \
  --prone-calibrate-on-start \
  --prone-pose down \
  --startup-ramp-seconds 12 \
  --prehold-seconds 2 \
  --duration 12 \
  --cmd-hz 100 \
  --mpc-hz 5 \
  --kp 50 \
  --kd 3 \
  --final-kp 25 \
  --final-kd 2.0 \
  --tau-limit 0.10 \
  --final-tau-limit 3.0 \
  --adaptive-tau-limit \
  --tau-limit-rate 0.4 \
  --adaptive-tilt-soft 0.025 \
  --adaptive-tilt-hard 0.070 \
  --abort-on-large-error \
  --abort-tilt 0.08 \
  --use-imu-base-state \
  --imu-gyro-deadband 0.005 \
  --fix-gateway-gyro-order \
  --mpc-r 1e-3 \
  --tau-limit-mode scale \
  --ramp-seconds 3 \
  --lift-leg FL \
  --lift-thigh-offset 0.08 \
  --lift-calf-offset -0.16 \
  --allow-single-leg-lift \
  --return-pose-on-exit down \
  --return-ramp-seconds 5 \
  --disable-on-exit \
  --allow-large-gains \
  --allow-large-tau-limit \
  --allow-large-gains \
  --robot-standing-supported \
  --i-accept-risk
```

日志新增字段：

```text
contact_mask
lift_leg
lift_alpha
```

`contact_mask` 顺序为：

```text
FL FR RL RR
```

例如 `0111` 表示 FL 被当作 no-contact，FR/RL/RR 三足支撑。

如果抬腿时侧倒或振荡：

```bash
--lift-thigh-offset 0.05
--lift-calf-offset -0.10
--final-tau-limit 2.5
--mpc-r 2e-3
```

## 17. EX34：real-state MPC 第一轮验证

EX34 用来验证“仿真 MPC 控制结构”的真机入口：

```text
lowstate + IMU + 支撑脚运动学
-> Pinocchio floating-base state
-> ComTraj + CentroidalMPC + LegController
-> 100Hz lowcmd q/kp/kd + tau feed-forward
```

第一轮只跑 `--gait all-stance`，不打开 trot/swing。

```bash
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

EX34 日志重点看：

```text
_summary.txt:
exit_reason
overlay_lowcmd_count
overlay_lowcmd_interval_max_s
publish_ms_max
state_ms_max
mpc_wall_ms_max

CSV:
base_source
height_source
vel_source
base_z
base_vx / base_vy / base_vz
base_roll / base_pitch
mpc_fx / mpc_fy / mpc_fz
tau_cmd
contact_mask
```
