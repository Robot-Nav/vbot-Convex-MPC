# VBot Real Affine

This folder is the single home for VBot real-joint affine work:

- `ex18_vbot_pose_viewer.py`: view/edit model poses.
- `ex19_vbot_solve_affine_calibration.py`: solve affine from stand/down captures.
- `ex20_vbot_solve_down_pose_ik.py`: solve a model down pose with IK.
- `ex26_vbot_generate_sign_bias_affine.py`: generate sign+bias affine maps.
- `ex27_vbot_generate_circular_down_bias_affine.py`: generate circular down-bias affine maps.
- `ex25_vbot_real_affine_bridge_check.py`: offline affine convention check.
- `EX23_vbot_real_joint_affine_test.py`: direct-serial real affine verifier.
- `EX24_vbot_pose_cycle_test.py`: direct-serial pose-cycle tester.
- `EX26_vbot_real_pose_pd.py`: DDS pose hold through the affine map.
- `ex21_vbot_affine_dds_test.py`: DDS affine monitor/single-joint test.
- `vbot_real_serial_utils.py`: shared direct-serial helpers.

Common commands:

```bash
python3 examples/vbot_real_affine/ex25_vbot_real_affine_bridge_check.py --no-pose
python3 examples/vbot_real_affine/EX23_vbot_real_joint_affine_test.py --help
```
