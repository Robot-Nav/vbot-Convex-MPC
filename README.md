# go2-convex-mpc / VBot MPC 项目分析

四足机器人凸模型预测控制（Convex MPC）控制栈，最初面向 Unitree Go2 开发，后适配 VBot 机器人。

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![MuJoCo](https://img.shields.io/badge/MuJoCo-2.3.3+-orange.svg)](https://github.com/google-deepmind/mujoco)
[![Pinocchio](https://img.shields.io/badge/Pinocchio-2.6.20+-green.svg)](https://github.com/stack-of-tasks/pinocchio)
[![CasADi](https://img.shields.io/badge/CasADi-3.6.3+-red.svg)](https://web.casadi.org/)
[![OSQP](https://img.shields.io/badge/OSQP-0.6.2+-yellow.svg)](https://osqp.org/)

---

**相关文章：**
- [四足机器人MPC控制：算法原理与实现](https://blog.csdn.net/qq_56908984/article/details/161836928?spm=1011.2415.3001.5331)
- [四足机器人MPC控制：项目结构与运行指南](https://blog.csdn.net/qq_56908984/article/details/161836868?spm=1011.2415.3001.5331)

---
sim2sim演示

https://github.com/user-attachments/assets/6c744851-63ee-4630-8fd7-0f04a32facda

原地踏步


https://github.com/user-attachments/assets/d9816d35-9ff2-4628-94db-f3e3b26c3b51


行走


sim2real演示




原地踏步



行走


> PS：原项目提出MPC求解总体过程约2.7ms，但后期sim2real测试中，整体控制循环包含了MPC求解、轨迹生成、状态更新以及通信延时等等，实机时间约60ms，约15hz，因此对源代码进行了代码优化以及参数调整，例如摆腿过程等，最后可达30-40hz，满足实时性；

---

## 1. 项目概述与意义

### 1.1 项目来源与目标

本项目是一个**四足机器人凸模型预测控制**（Convex MPC）控制栈，最初面向 **Unitree Go2** 四足机器人开发，后来在 **VBot** 四足机器人上进行了适配与部署验证。

项目核心目标是：

- 实现一套基于**接触力优化**的凸 MPC 控制器；
- 在 MuJoCo 仿真环境中验证四足机器人多种运动模态（原地踏步、前进、侧移、旋转、上楼梯）；
- 通过 Pinocchio 提供运动学、动力学、质心与足端雅可比计算；
- 将同一套 MPC 算法部署到真实 VBot 机器人硬件上。

> 一句话总结：**Pinocchio 负责"机器人身体计算"，MuJoCo 负责"物理世界仿真"，MPC 负责"未来接触力规划"。**

### 1.2 核心技术路线

控制算法参考 MIT Cheetah 3 的著名论文：

> **"Dynamic Locomotion in the MIT Cheetah 3 Through Convex Model-Predictive Control"**  
> https://dspace.mit.edu/bitstream/handle/1721.1/138000/convex_mpc_2fix.pdf

相比传统基于位置跟踪的四足控制，该方法的特点在于：

1. **直接优化地面接触力**，而非间接通过关节位置；
2. **使用质心动力学线性化模型**，将 MPC 问题转化为凸二次规划（QP）；
3. **每轮只执行预测序列的第一步**，滚动优化实现闭环鲁棒性；
4. **结合 Raibert 式足端落点规划与五次摆腿轨迹**，实现动态运动。

### 1.3 已验证能力

| 能力 | 指标 |
|------|------|
| 前进速度 | 最高 0.8 m/s（Go2 仿真） |
| 后退速度 | 最高 0.8 m/s |
| 侧向速度 | 最高 0.4 m/s |
| 旋转速度 | 最高 4.0 rad/s |
| 仿真步态 | Trot，3.0 Hz，duty cycle 0.6 |
| 控制循环 | 1000 Hz MuJoCo 物理 / 200 Hz 腿控 / 30–50 Hz MPC |

---

## 2. 系统架构

### 2.1 控制栈模块

```text
┌─────────────────────────────────────────────────────────────┐
│                     User Command (v_x, v_y, ω_z, z_h)        │
└───────────────────────┬─────────────────────────────────────┘
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              Reference Trajectory Generator (ComTraj)       │
│         生成未来 N 步的质心位置/姿态/速度/角速度参考           │
└───────────────────────┬─────────────────────────────────────┘
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                  Gait Scheduler (Gait)                       │
│              生成接触表 contact_table，判断支撑/摆动相         │
└───────────────────────┬─────────────────────────────────────┘
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                 Centroidal MPC (CentroidalMPC)               │
│           求解凸 QP，优化未来四足足端接触力序列               │
└───────────────────────┬─────────────────────────────────────┘
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                  Leg Controller (LegController)              │
│      支撑相：τ = J^T · (-F)      摆动相：阻抗+前馈跟踪足端轨迹  │
└───────────────────────┬─────────────────────────────────────┘
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              Robot Model (PinVBotModel / PinGo2Model)        │
│          运动学、质心、足端、雅可比、动力学项 M/C/g           │
└───────────────────────┬─────────────────────────────────────┘
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              Physics Sim / Real Robot (MuJoCo / DDS-CAN)     │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 核心频率

| 模块 | 频率 | 说明 |
|------|------|------|
| MuJoCo 物理积分 | 1000 Hz | 仿真时间推进 |
| 腿部控制器 | 200 Hz | 计算 12 关节力矩 |
| MPC 求解 | 30–50 Hz | 滚动优化接触力，每周期约 2.7 ms |
| Gait / 摆动轨迹 | 200 Hz | 判断相态、生成摆腿轨迹 |
| Viewer 刷新 | 60 Hz | 仅影响可视化 |

---

## 3. 算法原理

### 3.1 质心状态定义

机器人质心状态由 12 维向量描述：

$$\mathbf{x} = \left[\mathbf{p}^T,\ \boldsymbol{\Theta}^T,\ \mathbf{v}^T,\ \boldsymbol{\omega}^T\right]^T \in \mathbb{R}^{12}$$

其中：

- $\mathbf{p} = [x,\ y,\ z]^T$：质心在世界坐标系下的位置；
- $\boldsymbol{\Theta} = [\text{roll},\ \text{pitch},\ \text{yaw}]^T$：机体在世界坐标系下的 ZYX 欧拉角；
- $\mathbf{v} = [v_x,\ v_y,\ v_z]^T$：质心线速度；
- $\boldsymbol{\omega} = [\omega_x,\ \omega_y,\ \omega_z]^T$：机体角速度。

控制输入为四足各自的 3 维接触力，共 12 维：

$$\mathbf{u} = \left[\mathbf{f}_1^T,\ \mathbf{f}_2^T,\ \mathbf{f}_3^T,\ \mathbf{f}_4^T\right]^T \in \mathbb{R}^{12}$$

### 3.2 连续时间质心动力学

单刚体质心动力学方程为：

$$m \cdot \ddot{\mathbf{p}} = \sum \mathbf{f}_i - m \mathbf{g}$$
$$\mathbf{I} \cdot \dot{\boldsymbol{\omega}} = \sum (\mathbf{r}_i \times \mathbf{f}_i)$$

其中：

- $m$：机器人总质量；
- $\mathbf{I}$：机体在世界系下的惯量张量；
- $\mathbf{r}_i = \mathbf{p}_{\text{foot},i} - \mathbf{p}_{\text{com}}$：第 $i$ 条腿足端到质心的位置矢量；
- $\mathbf{g} = [0,\ 0,\ 9.81]^T$：重力加速度。

将姿态变化用欧拉角近似，且使用平均 yaw 角对应的 $R_z^T$ 将机体角速度映射到欧拉角速率，得到连续时间状态空间形式：

$$\dot{\mathbf{x}} = \mathbf{A}_c \mathbf{x} + \mathbf{B}_c \mathbf{u} + \mathbf{g}_c$$

其中：

$$\mathbf{A}_c = \begin{bmatrix}
\mathbf{0} & \mathbf{0} & \mathbf{I} & \mathbf{0} \\
\mathbf{0} & \mathbf{0} & \mathbf{0} & R_z^T \\
\mathbf{0} & \mathbf{0} & \mathbf{0} & \mathbf{0} \\
\mathbf{0} & \mathbf{0} & \mathbf{0} & \mathbf{0}
\end{bmatrix}$$

$$\mathbf{B}_c[i] = \begin{bmatrix}
\mathbf{0} & \mathbf{0} & \mathbf{0} & \mathbf{0} \\
\mathbf{0} & \mathbf{0} & \mathbf{0} & \mathbf{0} \\
\frac{1}{m}\mathbf{I} & \frac{1}{m}\mathbf{I} & \frac{1}{m}\mathbf{I} & \frac{1}{m}\mathbf{I} \\
\mathbf{I}^{-1}[\mathbf{r}_1]_\times & \mathbf{I}^{-1}[\mathbf{r}_2]_\times & \mathbf{I}^{-1}[\mathbf{r}_3]_\times & \mathbf{I}^{-1}[\mathbf{r}_4]_\times
\end{bmatrix}$$

$$\mathbf{g}_c = \left[0,\ 0,\ 0,\ 0,\ 0,\ 0,\ 0,\ 0,\ -g,\ 0,\ 0,\ 0\right]^T$$

这里 $[\mathbf{r}_i]_\times$ 表示 $\mathbf{r}_i$ 的反对称矩阵：

$$[\mathbf{r}]_\times = \begin{bmatrix}
0 & -r_z & r_y \\
r_z & 0 & -r_x \\
-r_y & r_x & 0
\end{bmatrix}$$

### 3.3 离散时间质心动力学

代码中使用解析欧拉离散化，而非矩阵指数，以提高计算速度。离散形式为：

$$\mathbf{x}_{k+1} = \mathbf{A}_d \mathbf{x}_k + \mathbf{B}_d[k] \mathbf{u}_k + \mathbf{g}_d$$

其中：

$$\mathbf{A}_d = \mathbf{I} + \mathbf{A}_c \cdot dt$$

具体非零块为：

- $\mathbf{A}_d[0:3,\ 6:9] = dt \cdot \mathbf{I}$：位置受速度影响；
- $\mathbf{A}_d[3:6,\ 9:12] = dt \cdot R_z^T$：欧拉角受角速度影响。

$$\mathbf{g}_d = \left[0.5 \cdot g \cdot dt^2,\ 0,\ g \cdot dt,\ 0\right]^T \quad (\text{仅位置、速度块})$$

$\mathbf{B}_d[k]$ 在代码中按时间步独立计算（因为足端杠杆 $\mathbf{r}_i(k)$ 随轨迹变化）：

$$\mathbf{B}_p = \left(\frac{0.5 \cdot dt^2}{m}\right) \cdot \mathbf{I}$$
$$\mathbf{B}_v = \left(\frac{dt}{m}\right) \cdot \mathbf{I}$$
$$\mathbf{W}_i = \mathbf{I}^{-1} \cdot [\mathbf{r}_i(k)]_\times$$

$$\mathbf{B}_d[k] = \begin{bmatrix}
\mathbf{B}_p & \mathbf{B}_p & \mathbf{B}_p & \mathbf{B}_p \\
0.5 dt^2 R_z^T \mathbf{W}_1 & \dots & 0.5 dt^2 R_z^T \mathbf{W}_4 \\
\mathbf{B}_v & \mathbf{B}_v & \mathbf{B}_v & \mathbf{B}_v \\
dt \mathbf{W}_1 & dt \mathbf{W}_2 & dt \mathbf{W}_3 & dt \mathbf{W}_4
\end{bmatrix}$$

### 3.4 模型预测控制（MPC）QP 问题

预测时域 $N = 16$，覆盖一个完整 Trot 步态周期。优化变量为：

$$\mathbf{w} = \left[\mathbf{x}_1^T,\ \dots,\ \mathbf{x}_N^T,\ \mathbf{u}_0^T,\ \dots,\ \mathbf{u}_{N-1}^T\right]^T \in \mathbb{R}^{N \cdot 12 + N \cdot 12}$$

目标函数为最小化参考轨迹偏差与输入代价：

$$\min \sum_{k=1}^N (\mathbf{x}_k - \mathbf{x}_k^{\text{ref}})^T \mathbf{Q} (\mathbf{x}_k - \mathbf{x}_k^{\text{ref}}) + \sum_{k=0}^{N-1} \mathbf{u}_k^T \mathbf{R} \mathbf{u}_k$$

写成标准 QP 形式：

$$\begin{aligned}
\min \quad & \frac{1}{2} \mathbf{w}^T \mathbf{H} \mathbf{w} + \mathbf{g}^T \mathbf{w} \\
\text{s.t.} \quad & \mathbf{A}_{\text{eq}} \mathbf{w} = \mathbf{b}_{\text{eq}} \\
& \mathbf{A}_{\text{ineq}} \mathbf{w} \leq \mathbf{b}_{\text{ineq}} \\
& \mathbf{lbx} \leq \mathbf{w} \leq \mathbf{ubx}
\end{aligned}$$

其中：

- $\mathbf{H}$：由 $\mathbf{Q}$ 和 $\mathbf{R}$ 组成的对角 Hessian 矩阵；
- $\mathbf{g} = -2 \mathbf{Q} \mathbf{x}_{\text{ref}}$：线性代价项；
- 等式约束为离散动力学约束：

$$\mathbf{x}_1 = \mathbf{A}_d \mathbf{x}_0 + \mathbf{B}_d[0] \mathbf{u}_0 + \mathbf{g}_d$$
$$\mathbf{x}_{k+1} = \mathbf{A}_d \mathbf{x}_k + \mathbf{B}_d[k] \mathbf{u}_k + \mathbf{g}_d, \quad k = 1,\ \dots,\ N-1$$

- 不等式约束为摩擦锥约束：

$$|f_x| \leq \mu \cdot f_z$$
$$|f_y| \leq \mu \cdot f_z$$

代码中实现为线性摩擦金字塔：

$$f_x - \mu f_z \leq 0$$
$$-f_x - \mu f_z \leq 0$$
$$f_y - \mu f_z \leq 0$$
$$-f_y - \mu f_z \leq 0$$

- Box 约束：
  - 摆动腿：三个方向力为 0；
  - 支撑腿：法向力满足 $f_z \in [f_{z,\text{min}},\ f_{z,\text{max}}]$。

### 3.5 求解器与实现优化

使用 **CasADi** 建模，底层调用 **OSQP** 求解稀疏 QP。

为达到实时性，代码做了以下优化：

1. **预计算常数矩阵**：Hessian $\mathbf{H}$、摩擦锥静态部分 $\mathbf{A}_{\text{ineq,static}}$、稀疏结构；
2. **参数化动力学函数**：`dyn_builder` 将 $\mathbf{A}_d$ 与 $\mathbf{B}_d$ 快速组装成块对角；
3. **Warm Start**：保留上一轮解 $\mathbf{x}_0$、$\lambda_{x,0}$、$\lambda_{a,0}$；
4. **稀疏矩阵增量更新**：每轮只更新时间相关部分（$\mathbf{B}_d$、参考轨迹、接触表、边界）。

典型求解耗时：

- QP 矩阵更新：约 1.0 ms；
- OSQP 求解：约 1.7 ms；
- 单轮总 MPC 时间：约 2.7 ms（可支持 48 Hz 以上控制）。

---

## 4. 步态与落点规划

### 4.1 接触表生成

Trot 步态使用四足相位偏移：

```python
PHASE_OFFSET = [0.5, 0.0, 0.0, 0.5]   # FL, FR, RL, RR
```

第 $i$ 条腿在时刻 $t$ 的相位为：

$$\phi_i(t) = \text{mod}(\phi_{\text{offset},i} + t / T_{\text{gait}},\ 1.0)$$

当 $\phi_i < \text{duty}$ 时该腿为支撑相（stance），否则为摆动相（swing）。

### 4.2 Raibert 风格足端落点

摆动腿离地瞬间，根据当前机体速度、期望速度、yaw 角速度预测下一次触地点：

$$\mathbf{p}_{\text{td}} = \mathbf{p}_{\text{hip}} + \mathbf{v}_{\text{des}} \cdot T_{\text{pred}} + k_p \cdot (\mathbf{p}_{\text{com}} - \mathbf{p}_{\text{des}}) + k_v \cdot (\mathbf{v}_{\text{com}} - \mathbf{v}_{\text{des}}) + \mathbf{p}_{\text{rot}}$$

其中：

- $\mathbf{p}_{\text{hip}}$：髋关节在世界系下水平位置；
- $T_{\text{pred}} = (t_{\text{swing}} + 0.5 \cdot t_{\text{stance}}) / 2$：预测触地时间；
- 旋转修正项 $\mathbf{p}_{\text{rot}}$ 补偿机体 yaw 运动造成的足端切向偏移；
- $k_p,\ k_v$ 为小增益，用于修正速度/位置跟踪误差。

### 4.3 摆腿轨迹

摆动相使用**最小加加速度（minimum-jerk）五次多项式**连接起点 $\mathbf{p}_0$ 和触地点 $\mathbf{p}_f$：

$$s = t / T_{\text{swing}} \in [0,\ 1]$$
$$\mathbf{p}(s) = \mathbf{p}_0 + (\mathbf{p}_f - \mathbf{p}_0) \cdot (10 s^3 - 15 s^4 + 6 s^5)$$

在竖直方向叠加平滑凸起：

$$b(s) = 64 s^3 (1 - s)^3$$
$$p_z(s) += h_{\text{sw}} \cdot b(s)$$

其中 $h_{\text{sw}}$ 为摆腿最大高度。该凸起在起点和终点处速度、加速度均为 0，保证触地平滑。

---

## 5. 腿部控制器

### 5.1 支撑相（Stance）

支撑腿将 MPC 优化出的接触力通过足端雅可比映射为关节力矩：

$$\boldsymbol{\tau} = \mathbf{J}_{\text{foot}}^T \cdot (-\mathbf{F}_{\text{MPC}})$$

负号表示 MPC 优化的 $\mathbf{F}$ 是地面作用于脚的力，而脚需要给地面以反作用力。

### 5.2 摆动相（Swing）

摆动腿采用操作空间阻抗控制：

$$\mathbf{F} = \mathbf{K}_p \cdot (\mathbf{p}_{\text{des}} - \mathbf{p}_{\text{now}}) + \mathbf{K}_d \cdot (\mathbf{v}_{\text{des}} - \mathbf{v}_{\text{now}}) + \boldsymbol{\Lambda} \cdot (\mathbf{a}_{\text{des}} - \dot{\mathbf{J}} \dot{\mathbf{q}})$$
$$\boldsymbol{\tau} = \mathbf{J}_{\text{foot}}^T \cdot \mathbf{F} + (\mathbf{C} \dot{\mathbf{q}} + \mathbf{g})_{\text{leg}}$$

其中：

- $\mathbf{K}_p = \text{diag}(400,\ 400,\ 400)$；
- $\mathbf{K}_d = \text{diag}(75,\ 75,\ 75)$；
- $\boldsymbol{\Lambda} = (\mathbf{J} \mathbf{M}^{-1} \mathbf{J}^T)^{-1}$：操作空间惯量；
- $\mathbf{a}_{\text{des}}$ 为期望足端加速度；
- 最后加入关节空间偏置项 $\mathbf{C} \dot{\mathbf{q}} + \mathbf{g}$ 补偿重力与科氏力。

---

## 6. 项目结构与内容

### 6.1 目录结构

```text
go2-convex-mpc/
├── src/convex_mpc/                 # 核心控制算法包
│   ├── centroidal_mpc.py           # 凸 MPC 求解器
│   ├── com_trajectory.py           # 质心参考轨迹生成 + 线性化动力学
│   ├── gait.py                     # 步态调度 + 落点 + 摆腿轨迹
│   ├── leg_controller.py           # 腿部力矩控制器
│   ├── go2_robot_data.py           # Go2 Pinocchio 模型接口
│   ├── vbot_robot_data.py          # VBot Pinocchio 模型接口
│   ├── mujoco_model.py             # Go2 MuJoCo 接口
│   ├── mujoco_vbot_model.py        # VBot MuJoCo 接口
│   ├── vbot_real_affine.py         # 真实电机与模型关节的仿射变换
│   └── plot_helper.py              # 结果绘图工具
│
├── examples/
│   ├── vbot_simulation_mpc/        # VBot MuJoCo 仿真示例
│   │   ├── ex16_vbot_mpc_trot_in_place.py
│   │   ├── ex17_vbot_mpc_keyboard_control.py
│   │   ├── ex18_vbot_mpc_stairs_keyboard_control.py
│   │   └── ex23_vbot_ex35_forward_walk_sim.py
│   ├── real_mpc_experiments/       # 真实 VBot 机器人实验脚本
│   │   ├── EX29_vbot_dds_lowstate_monitor.py
│   │   ├── EX30_vbot_real_mpc_monitor.py
│   │   ├── EX31_vbot_real_mpc_torque_test.py
│   │   ├── EX33B_vbot_dds_stand_mpc_overlay.py
│   │   └── EX34_vbot_dds_real_mpc_state_estimator.py
│   └── vbot_real_affine/           # 真实电机标定与测试
│
├── models/
│   ├── URDF/go2_description/       # Go2 URDF 模型
│   ├── MJCF/go2/                   # Go2 MuJoCo 场景
│   └── MJCF/vbot/                  # VBot MuJoCo / Pinocchio 模型
│
├── configs/                        # YAML 配置文件
│   ├── ex34_forward_walk_slow_imu.yaml
│   ├── ex34_trot_final_fast_imu.yaml
│   ├── ex36_trot_mpc_swing_force_slow.yaml
│   └── vbot_real_joint_affine.yaml
│
├── environment.yml                 # Conda 环境配置
├── pyproject.toml                  # Python 包配置
├── README.md                       # 英文项目说明
├── VBOT_MPC_WORKSPACE_NOTES.md     # VBot 工作空间说明
└── TECHNICAL_DOCUMENTATION_ZH.md   # 本技术文档
```

### 6.2 核心文件说明

| 文件 | 作用 |
|------|------|
| `src/convex_mpc/centroidal_mpc.py` | 构建并求解 MPC QP，管理 OSQP 求解器、稀疏矩阵、warm start。 |
| `src/convex_mpc/com_trajectory.py` | 根据速度指令生成参考轨迹，计算 $\mathbf{A}_d$、$\mathbf{B}_d[k]$、$\mathbf{g}_d$。 |
| `src/convex_mpc/gait.py` | 生成 contact table、Raibert 落点、minimum-jerk 摆腿轨迹。 |
| `src/convex_mpc/leg_controller.py` | 支撑相力矩映射、摆动相阻抗控制。 |
| `src/convex_mpc/vbot_robot_data.py` | VBot Pinocchio 模型接口，含运动学、雅可比、质心、动力学。 |
| `src/convex_mpc/mujoco_vbot_model.py` | VBot MuJoCo 接口，状态同步与力矩下发。 |
| `src/convex_mpc/vbot_real_affine.py` | 真实电机反馈与模型关节之间的 scale/bias 仿射变换。 |

### 6.3 坐标系与向量约定

- **世界坐标系**：$x$ 前，$y$ 左，$z$ 上；
- **机体坐标系**：与机体固连，base 坐标；
- **力矩/关节顺序**（MPC 内部）：`[FL, FR, RL, RR]`，每条腿 `[hip, thigh, calf]`；
- **DDS/CAN 真机顺序**：`[FR, FL, RR, RL]`，由 `vbot_real_affine.py` 负责转换。

---

## 7. 运行步骤

### 7.1 环境安装

> 推荐 Linux 环境，其它操作系统未测试。

1. 克隆仓库：

```bash
git clone https://github.com/elijah-waichong-chan/go2-convex-mpc.git
cd go2-convex-mpc
```

2. 创建 Conda 环境：

```bash
conda env create -f environment.yml
conda activate go2-convex-mpc
```

如果提示包导入错误，执行：

```bash
pip install -e .
```

3. 导入检查：

```bash
export PYTHONNOUSERSITE=1
python - <<'PY'
import mujoco, pinocchio, casadi, convex_mpc
print("mujoco:", mujoco.__version__)
print("pinocchio:", pinocchio.__version__)
print("casadi:", casadi.__version__)
print("convex_mpc: OK")
PY
```

### 7.2 仿真示例（VBot）

#### 原地 Trot

```bash
python3 examples/vbot_simulation_mpc/ex16_vbot_mpc_trot_in_place.py
```

回放：

```bash
VBOT_MPC_REPLAY=1 python3 examples/vbot_simulation_mpc/ex16_vbot_mpc_trot_in_place.py
```

绘图：

```bash
VBOT_MPC_PLOTS=1 python3 examples/vbot_simulation_mpc/ex16_vbot_mpc_trot_in_place.py
```

#### 键盘控制

```bash
python3 examples/vbot_simulation_mpc/ex17_vbot_mpc_keyboard_control.py
```

按键：

| 键 | 功能 |
|----|------|
| W/S | 增加/减小前进速度 |
| A/D | 增加/减小侧向速度 |
| Q/E | 增加/减小 yaw 角速度 |
| Space | 停止 |
| R | 重置 |
| Esc | 退出 |

#### 前进走仿真（EX35 真机逻辑镜像）

```bash
python3 examples/vbot_simulation_mpc/ex23_vbot_ex35_forward_walk_sim.py \
  --duration 6 \
  --x-vel 0.03
```

带可视化：

```bash
python3 examples/vbot_simulation_mpc/ex23_vbot_ex35_forward_walk_sim.py \
  --duration 8 \
  --x-vel 0.03 \
  --viewer \
  --realtime
```

#### 楼梯键盘控制

```bash
python3 examples/vbot_simulation_mpc/ex18_vbot_mpc_stairs_keyboard_control.py
```

### 7.3 真实 VBot 机器人实验

> 真机实验存在安全风险，必须按顺序逐步验证。

推荐调试顺序：

1. `EX29_vbot_dds_lowstate_monitor.py`：只读 lowstate，验证通信与反馈映射；
2. `EX30_vbot_real_mpc_monitor.py`：运行 MPC 但不发送力矩；
3. `EX31_vbot_real_mpc_torque_test.py`：悬空低力矩测试；
4. `EX33B_vbot_dds_stand_mpc_overlay.py`：站立 MPC overlay；
5. `EX34_vbot_dds_real_mpc_state_estimator.py`：完整 MPC 状态估计 + all-stance；
6. 确认稳定后再开启 trot/swing。

#### 只读监控（EX29）

```bash
python3 examples/real_mpc_experiments/EX29_vbot_dds_lowstate_monitor.py \
  --network lo \
  --duration 5 \
  --prone-calibrate-on-start
```

#### MPC 监控但不使能（EX30）

```bash
python3 examples/real_mpc_experiments/EX30_vbot_real_mpc_monitor.py \
  --duration 10 \
  --prone-calibrate-on-start
```

#### 悬空低力矩测试（EX31）

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

#### 站立 MPC overlay（EX33B）

```bash
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
  --kp 50 --kd 3 \
  --final-kp 30 --final-kd 2.5 \
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
  --robot-standing-supported \
  --i-accept-risk
```

#### 真实状态 MPC（EX34）

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
  --kp 50 --kd 3 \
  --final-kp 50 --final-kd 3 \
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

> 注意：所有真机命令都需要显式传入 `--i-accept-risk` 以确认风险。

---

## 8. 关键参数说明

### 8.1 MPC 权重

定义在 `src/convex_mpc/centroidal_mpc.py`：

```python
COST_MATRIX_Q = diag([1, 1, 50, 10, 20, 1, 2, 2, 1, 1, 1, 1])
COST_MATRIX_R = diag([1e-5] * 12)
```

对应状态顺序 `[x, y, z, roll, pitch, yaw, vx, vy, vz, ωx, ωy, ωz]`。

- $\mathbf{Q}[2] = 50$：高度权重最大，优先保持机身高度；
- $\mathbf{Q}[4] = 20$：pitch 权重较高，防止机身前后倾倒；
- $\mathbf{R}$ 很小：允许接触力较大变化，保证动态响应。

### 8.2 摩擦与地面接触

```python
MU = 0.8          # 摩擦系数
FZ_MIN = 10.0     # 支撑腿最小法向力
FZ_MAX = inf      # 支撑腿最大法向力（真机部署前需限制）
```

### 8.3 步态参数

```python
GAIT_HZ = 3.0
GAIT_DUTY = 0.6
PHASE_OFFSET = [0.5, 0.0, 0.0, 0.5]   # Trot
HEIGHT_SWING = 0.1                    # 摆腿高度
```

### 8.4 摆动腿阻抗

```python
KP_SWING = diag([400, 400, 400])
KD_SWING = diag([75, 75, 75])
```

### 8.5 VBot 仿真力矩限幅

```python
TAU_LIM = 0.9 * [17, 17, 34] * 4 legs
```

即每条腿 hip/thigh $\pm$ 15.3 Nm，calf $\pm$ 30.6 Nm。

---

## 9. 仿真到真机的关键桥接

### 9.1 电机-模型仿射变换

真实电机编码器读数 $q_{\text{motor}}$ 与模型关节 $q_{\text{model}}$ 之间存在 scale 与 bias：

$$q_{\text{model}} = \text{scale} \cdot q_{\text{motor}} - \text{bias}$$

由虚功原理，力矩映射为：

$$\tau_{\text{motor}} = \text{scale} \cdot \tau_{\text{model}}$$

相关配置见 `configs/vbot_real_joint_affine.yaml`。

### 9.2 状态估计

真机版本 `EX34` 通过以下信息估计 floating-base 状态：

- IMU roll/pitch/gyro；
- 关节电机反馈（位置、速度，部分情况含力矩）；
- 支撑脚运动学约束估计 base 高度与水平速度。

模式由参数控制：

```bash
--base-height-mode stance-feet   # 用支撑脚估计高度
--base-vel-mode stance-feet      # 用支撑脚估计水平速度
--use-imu-base-state             # 使用 IMU 作为 base 姿态与角速度
```

### 9.3 安全机制

真机脚本内置多层保护：

- 启动/退出时姿态斜坡过渡；
- 力矩自适应限幅（根据机身 tilt 调整）；
- 大关节误差、大机身倾角自动 abort；
- 退出时自动回到趴下姿态并 disable 电机；
- 必须显式 `--i-accept-risk` 才会发送非零力矩。

---

## 10. 调试建议

### 10.1 仿真阶段优先调节参数

如果仿真中出现不稳定，建议按以下顺序调整：

1. **降低速度限幅**（`X_VEL_LIMIT`、`Y_VEL_LIMIT`、`YAW_RATE_LIMIT`）；
2. **调整期望高度** `DEFAULT_Z_POS`；
3. **调整步态频率** `GAIT_HZ` 与 duty；
4. **调整 MPC 权重** $\mathbf{Q}$、$\mathbf{R}$；
5. **调整摆腿高度** `HEIGHT_SWING` 与阻抗 $\mathbf{K}_p$/$\mathbf{K}_d$；
6. **检查摩擦系数** $\mu$ 是否与 MuJoCo 场景一致。

### 10.2 实物部署前 checklist

- 电机方向、零点、力矩方向已验证；
- 仿射标定文件 `vbot_real_joint_affine.yaml` 已标定；
- 急停、通信超时、力矩限幅、关节限位保护已启用；
- 先在悬空/吊绳/保护架下测试；
- 从 all-stance 站立开始，再逐步开启 swing；
- 日志CSV与 summary 已配置，便于离线分析。

---

## 11. 参考与致谢

- MIT Cheetah 3 Convex MPC 论文：  
  https://dspace.mit.edu/bitstream/handle/1721.1/138000/convex_mpc_2fix.pdf
- MuJoCo：https://github.com/google-deepmind/mujoco
- Pinocchio：https://github.com/stack-of-tasks/pinocchio
- CasADi：https://web.casadi.org/
- OSQP：https://osqp.org/

本项目最初作为 UC Berkeley MEng 机械工程顶点项目开发，后续适配至 VBot 真实四足机器人平台。

---

## 致谢

本项目基于 [go2-convex-mpc](https://github.com/elijah-waichong-chan/go2-convex-mpc) 开发，感谢原作者的开源贡献。


Uploading 录屏 2026-06-09 11-55-21.mp4…

