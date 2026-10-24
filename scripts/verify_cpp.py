"""Compile portable inference and compare full/head outputs with golden vectors."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess


def verify(project, build_dir, compiler=None, artifact_dir=None, model_data_dir=None):
    project = Path(project).resolve()
    build_dir = Path(build_dir).resolve()
    build_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = Path(artifact_dir or project / "artifacts").resolve()
    model_data_dir = Path(model_data_dir or project / "model_data").resolve()
    compiler = compiler or os.environ.get("CXX") or shutil.which("g++")
    if not compiler:
        raise RuntimeError("A C++17 g++ compiler is required for native verification.")
    extension = ".exe" if os.name == "nt" else ""
    full = build_dir / ("model_runner" + extension)
    head = build_dir / ("head_runner" + extension)
    common = [compiler, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
              "-Wpedantic", "-fno-exceptions", "-fno-rtti",
              "-I", str(project / "firmware"), "-I", str(model_data_dir)]
    subprocess.run(common + [str(project / "firmware/model_runtime.cpp"),
                             str(project / "firmware/host_runner.cpp"),
                             "-o", str(full)], check=True)
    subprocess.run(common + [str(project / "firmware/head_runtime.cpp"),
                             str(project / "firmware/head_runner.cpp"),
                             "-o", str(head)], check=True)
    golden_dir = artifact_dir / "golden"
    vectors = json.loads((golden_dir / "expected.json").read_text(encoding="utf-8"))
    if len(vectors) < 4:
        raise RuntimeError("Expected at least four independent class examples.")
    records = []
    for vector in vectors:
        for variant, executable, input_key in (
            ("full", full, "image_file"), ("head", head, "feature_file")):
            result = subprocess.run(
                [str(executable), str(golden_dir / vector[input_key])],
                text=True, capture_output=True, check=True,
            )
            output = json.loads(result.stdout)
            if output["class_id"] != vector["expected_class"] or output["logits"] != vector["expected_logits"]:
                raise AssertionError(f"{variant} output differs for {vector[input_key]}: {output}")
            if variant == "full" and output["arena_bytes"] != 38448:
                raise AssertionError("Unexpected activation buffer layout.")
            records.append({"variant": variant, "input": vector[input_key], "matched": True})
    report = {
        "target": "Native host C++17, board identity not verified",
        "vectors_per_variant": len(vectors),
        "exact_logit_comparisons": len(records),
        "all_bit_exact": True,
        "activation_arena_sizeof": 38448,
        "records": records,
        "limitations": ["UNO Q MCU compilation, total board memory, Bridge transport, and HID remain unverified."],
    }
    (artifact_dir / "native_verification.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: report[key] for key in (
        "target", "vectors_per_variant", "exact_logit_comparisons", "all_bit_exact"
    )}))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", default="build/native")
    parser.add_argument("--compiler")
    parser.add_argument("--artifact-dir")
    parser.add_argument("--model-data-dir")
    args = parser.parse_args()
    verify(Path(__file__).resolve().parent.parent, args.build_dir, args.compiler,
           args.artifact_dir, args.model_data_dir)


if __name__ == "__main__":
    main()
