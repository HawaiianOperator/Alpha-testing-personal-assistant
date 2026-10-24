"""Compile and verify persistent classification, framing, and owned-buffer clearing."""
import argparse
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import time


def frame(mode, task_id, payload):
    return struct.pack("<4sB3xII", b"PAM1", mode, task_id, len(payload)) + payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", default="build/native")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    build = Path(args.build_dir).resolve()
    build.mkdir(parents=True, exist_ok=True)
    compiler = os.environ.get("CXX") or shutil.which("g++")
    if not compiler:
        raise SystemExit("g++ supporting C++17 is required.")
    executable = build / ("classifier_service.exe" if os.name == "nt" else "classifier_service")
    subprocess.run([compiler, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror", "-Wpedantic",
                    "-fno-exceptions", "-fno-rtti", "-I", str(root / "firmware"),
                    "-I", str(root / "model_data")] +
                   [str(root / "firmware" / source) for source in (
                       "model_runtime.cpp", "head_runtime.cpp", "task_state.cpp", "classifier_service.cpp")] +
                   ["-o", str(executable)], check=True)
    golden = root / "artifacts/golden"
    vectors = json.loads((golden / "expected.json").read_text())
    packets, expected = [], []
    job_id = 0
    for vector in vectors:
        for mode, key in ((1, "image_file"), (2, "feature_file")):
            job_id += 1
            packets.append(frame(mode, job_id, (golden / vector[key]).read_bytes()))
            expected.append((job_id, vector["expected_class"], vector["expected_logits"]))
    started = time.monotonic()
    result = subprocess.run([str(executable)], input=b"".join(packets),
                            capture_output=True, check=True, timeout=30)
    responses = [json.loads(line) for line in result.stdout.splitlines()]
    if len(responses) != len(expected):
        raise AssertionError("Response count mismatch.")
    for response, wanted in zip(responses, expected):
        if (response["task_id"], response["class_id"], response["logits"]) != wanted:
            raise AssertionError("Persistent service differs from reference.")
        if response.get("buffers_cleared") is not True:
            raise AssertionError("Owned buffers were not cleared.")
    malformed = [b"truncated", struct.pack("<4sB3xII", b"PAM1", 1, 1, 999999),
                 frame(2, 0, bytes(16)), frame(2, 1, bytes([255]) * 16)]
    for packet in malformed:
        rejected = subprocess.run([str(executable)], input=packet, capture_output=True, timeout=10)
        if rejected.returncode == 0 or rejected.stdout:
            raise AssertionError("Malformed request was not rejected.")
    report = {"target": "Native host; not UNO Q hardware",
              "requests_in_one_process": len(responses),
              "all_outputs_match": True, "owned_buffers_cleared_per_request": True,
              "malformed_inputs_rejected": len(malformed),
              "host_batch_seconds": time.monotonic() - started,
              "scope": "Visual classification only; no general task planning or input execution."}
    (root / "artifacts/service_verification.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
