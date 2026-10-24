# Verify the trained classifier on Arduino UNO Q Linux

Use the updated project source with your own exported artifacts_jupyter folder. Do not replace trained artifacts with supplied demonstration weights.

In the GPU VM notebook after INT8 export:

```python
import subprocess, sys
subprocess.run([sys.executable, "scripts/package_uno_q.py", "--artifact-dir", "artifacts_jupyter", "--out", "uno-q-runtime.zip"], cwd=PROJECT_ROOT, check=True)
```

Download uno-q-runtime.zip from Jupyter and transfer it to UNO Q Linux using your existing board connection. This package is not a GPU VM executable: it compiles natively on the board. No CUDA or NumPy is needed for the verification runner. Python 3 and g++ supporting C++17 must be installed.

In a terminal ON THE BOARD:

```sh
python3 -m zipfile -e uno-q-runtime.zip .
cd uno-q-runtime
python3 scripts/run_uno_q.py
```

Success reports all_checks_passed=true. The resulting build/native/classifier_service is a binary-framed classifier process, not an interactive text assistant. It accepts the protocol documented in UNO_Q.md. The runner verifies ARM64 Linux, not board identity; run it on the actual UNO Q. It writes synthetic validation reports and build outputs, not private task captures. Board-wide swap/log settings remain a separate audit. MCU flashing, Bridge and desktop control remain unimplemented.
