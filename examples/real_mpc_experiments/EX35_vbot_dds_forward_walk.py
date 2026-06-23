#!/usr/bin/env python3
"""EX35: slow forward-walk launcher for EX34 real MPC.

This wrapper keeps the real control implementation in EX34 and only selects
the forward-walk YAML defaults. Extra command-line flags still override YAML
values, for example:

    python3 examples/real_mpc_experiments/EX35_vbot_dds_forward_walk.py \
      --i-accept-risk --x-vel 0.02
"""

from __future__ import annotations

import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO / "configs" / "ex34_forward_walk_slow_imu.yaml"


def _has_option(argv, option):
    prefix = option + "="
    return any(arg == option or arg.startswith(prefix) for arg in argv)


def main() -> int:
    if not _has_option(sys.argv[1:], "--config"):
        sys.argv[1:1] = ["--config", str(DEFAULT_CONFIG)]

    from EX34_vbot_dds_real_mpc_state_estimator import main as ex34_main

    return ex34_main()


if __name__ == "__main__":
    raise SystemExit(main())
