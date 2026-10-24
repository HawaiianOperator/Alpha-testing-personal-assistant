"""Package a selected trained export for native verification on UNO Q Linux."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def package(root, artifacts, output):
    root, artifacts, output = Path(root).resolve(), Path(artifacts).resolve(), Path(output).resolve()
    files = {}
    headers = artifacts / "model_data"
    if artifacts == root / "artifacts" and not headers.is_dir():
        headers = root / "model_data"
    for folder in ("firmware",):
        for p in (root / folder).rglob("*"):
            if p.is_file(): files[p.relative_to(root).as_posix()] = p
    for name in ("verify_cpp.py", "verify_lifecycle.py", "verify_service.py", "run_uno_q.py"):
        files["scripts/" + name] = root / "scripts" / name
    for name in ("model_data.h", "model_head.h"):
        files["model_data/" + name] = headers / name
    for name in ("model_int8.npz", "model_encoder.npz", "model_head.npz", "quantization_report.json"):
        files["artifacts/" + name] = artifacts / name
    for p in (artifacts / "golden").glob("*"):
        if p.is_file(): files["artifacts/golden/" + p.name] = p
    if "artifacts/golden/expected.json" not in files:
        raise FileNotFoundError("Run INT8 export first: golden vectors are missing.")
    for p in files.values():
        if not p.is_file(): raise FileNotFoundError(str(p))
    manifest = {name: hashlib.sha256(p.read_bytes()).hexdigest() for name, p in files.items()}
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, p in files.items(): archive.write(p, "uno-q-runtime/" + name)
        archive.writestr("uno-q-runtime/manifest.json", json.dumps(manifest, indent=2))
    print("Created:", output)
    print("Contains selected weights, matching headers/goldens, and native source; no private captures.")
    return output

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", default="artifacts_jupyter")
    parser.add_argument("--out", default="uno-q-runtime.zip")
    args = parser.parse_args()
    package(Path(__file__).resolve().parent.parent, args.artifact_dir, args.out)
