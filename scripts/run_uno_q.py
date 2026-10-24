"""Verify package integrity and compile/test on ARM64 Linux; never flashes MCU."""
import argparse
import hashlib
import json
import platform
from pathlib import Path
import shutil
import subprocess
import sys

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-host", action="store_true", help="Host-only test; does not establish UNO Q deployment.")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    arm_linux = platform.system() == "Linux" and platform.machine().lower() in ("aarch64", "arm64")
    if not arm_linux and not args.allow_host:
        raise SystemExit("Run on UNO Q ARM64 Linux, or use --allow-host for a host-only check.")
    if not shutil.which("g++"):
        raise SystemExit("Missing g++ C++17 compiler. Install the board's development tools first.")
    manifest = json.loads((root / "manifest.json").read_text())
    for name, wanted in manifest.items():
        p = (root / name).resolve()
        if root not in p.parents or hashlib.sha256(p.read_bytes()).hexdigest() != wanted:
            raise RuntimeError("Package integrity failure: " + name)
    for script in ("verify_cpp.py", "verify_lifecycle.py", "verify_service.py"):
        subprocess.run([sys.executable, str(root / "scripts" / script)], cwd=root, check=True)
    report = {"system": platform.system(), "machine": platform.machine(),
              "arm64_linux": arm_linux, "board_identity_verified": False,
              "all_checks_passed": True,
              "scope": "Native full/head logits, lifecycle, and classifier service; no MCU flashing, Bridge, or desktop control."}
    (root / "artifacts/target_verification.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
