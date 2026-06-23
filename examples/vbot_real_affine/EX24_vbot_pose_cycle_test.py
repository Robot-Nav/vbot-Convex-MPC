"""EX24: Direct-serial VBot smooth pose-cycle tester.

Adds ``pose-cycle`` mode on top of the EX23 mapping verifier.  Typical use:
smoothly move selected legs from the measured pose to ``down`` and then to
``stand`` while using the same real-joint affine map.
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
        description="VBot real joint mapping and pose-cycle tester using direct serial",
        fatu_serial_path=FATUDOG_SERIAL,
        include_pose_cycle=True,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    validate_serial_args(args, include_pose_cycle=True)
    return run_serial_mapping_tool(
        args,
        fatu_serial_path=FATUDOG_SERIAL,
        include_pose_cycle=True,
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
