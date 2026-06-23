"""EX23: Direct-serial VBot real-joint affine mapping verifier.

This entry validates and tests the real motor angle to model joint angle mapping:

    q_model = scale * q_motor - bias
    dq_model = scale * dq_motor
    q_motor_cmd = (q_model_cmd + bias) / scale

Use this for stand/down two-point affine calibration and single-leg pose tests.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[2]
WORKSPACE = REPO.parent
DEFAULT_FATUDOG_SERIAL = WORKSPACE / "fatuDog" / "serial_dds_gateway"


def default_fatudog_serial() -> str:
    env_path = os.environ.get("FATUDOG_SERIAL")
    if env_path:
        return env_path
    candidates = (
        DEFAULT_FATUDOG_SERIAL,
        WORKSPACE / "fatuDog0609" / "fatuDog" / "serial_dds_gateway",
    )
    for path in candidates:
        if path.exists():
            return str(path)
    return str(DEFAULT_FATUDOG_SERIAL)


_pre_parser = argparse.ArgumentParser(add_help=False)
_pre_parser.add_argument(
    "--fatudog-serial",
    default=default_fatudog_serial(),
)
_pre_args, _ = _pre_parser.parse_known_args()
FATUDOG_SERIAL = Path(_pre_args.fatudog_serial).expanduser().resolve()

if str(FATUDOG_SERIAL) not in sys.path:
    sys.path.insert(0, str(FATUDOG_SERIAL))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from vbot_real_serial_utils import (  # noqa: E402
    build_serial_arg_parser,
    run_serial_mapping_tool,
    validate_serial_args,
)


def parse_args():
    parser = build_serial_arg_parser(
        description="VBot real joint mapping verifier using direct serial",
        fatu_serial_path=FATUDOG_SERIAL,
        include_pose_cycle=False,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    validate_serial_args(args, include_pose_cycle=False)
    return run_serial_mapping_tool(
        args,
        fatu_serial_path=FATUDOG_SERIAL,
        include_pose_cycle=False,
        sign_only=False,
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
