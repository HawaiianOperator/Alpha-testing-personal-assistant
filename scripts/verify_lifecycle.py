"""Compile C++ inference and owned-buffer erasure self-tests; no device access."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", default="build/native")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    build = Path(args.build_dir).resolve()
    build.mkdir(parents=True, exist_ok=True)
    compiler = os.environ.get("CXX") or shutil.which("g++")
    if not compiler:
        raise SystemExit("A g++ compiler supporting C++17 is required.")
    tests = [
        ("runtime_self_test", ["model_runtime.cpp", "self_test.cpp"]),
        ("head_self_test", ["head_runtime.cpp", "head_self_test.cpp"]),
        ("task_state_self_test", ["task_state.cpp", "task_state_self_test.cpp"]),
    ]
    results = []
    for name, files in tests:
        executable = build / (name + (".exe" if os.name == "nt" else ""))
        subprocess.run([compiler, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
                        "-Wpedantic", "-fno-exceptions", "-fno-rtti"] +
                       [str(root / "firmware" / file) for file in files] +
                       ["-o", str(executable)], check=True)
        output = subprocess.run([str(executable)], check=True, capture_output=True, text=True)
        results.append({"test": name, "passed": True, "output": output.stdout.strip()})
    report = {"target": "Native host C++17; board identity not verified",
              "all_passed": True, "tests": results,
              "scope": "Explicit owned buffers only; no claim covers OS/SDK copies or physical remanence."}
    (root / "artifacts/lifecycle_verification.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
