# VBot Simulation MPC

MuJoCo/Pinocchio VBot MPC simulation examples.

- `ex16_vbot_mpc_trot_in_place.py`: flat-ground trot-in-place validation.
- `ex17_vbot_mpc_keyboard_control.py`: flat-ground keyboard velocity control.
- `ex18_vbot_mpc_stairs_keyboard_control.py`: known-stair terrain keyboard control.
- `ex23_vbot_ex35_forward_walk_sim.py`: MuJoCo simulation mirror of the
  EX35/EX34 real forward-walk mixed PD + MPC tau_ff logic.

`ex18` writes CSV runtime logs to `examples/vbot_simulation_mpc/logs/`.
Watch `base_x`, `base_z`, `roll`, `pitch`, `tau_max`, and `tau_sat_frac`
when tuning speed and stair parameters.

Run the EX35-style forward walk simulation without a viewer:

```bash
python3 examples/vbot_simulation_mpc/ex23_vbot_ex35_forward_walk_sim.py \
  --duration 6 \
  --x-vel 0.03
```

Open the MuJoCo viewer:

```bash
python3 examples/vbot_simulation_mpc/ex23_vbot_ex35_forward_walk_sim.py \
  --duration 8 \
  --x-vel 0.03 \
  --viewer \
  --realtime
```

The script loads `configs/ex34_forward_walk_slow_imu.yaml` by default and
accepts command-line overrides such as `--x-vel -0.02`, `--mpc-hz 5`, and
`--mpc-force-xy-scale 0.0`.
