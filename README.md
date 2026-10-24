# Personal Assistant: from-scratch model prototype

**Current hardware target: Arduino UNO Q 4 GB / 32 GB.** The previous ESP32-WROVER specification is superseded.

This package contains a trained visual-state model and full/partial integer inference. It is **not a finished general-language desktop agent**. Explicit Google/Search/Compute routing, keyword-prefix autocomplete, and current-task clarification are implemented. A separate text intent model trained from scratch now supports bounded search-link and arithmetic requests, clarification, and rejection. See TEXT_MODEL.md. General-language completion, unrestricted task planning, calendar scheduling, search-result analysis, and physical input execution remain unimplemented. See [COMMANDS.md](COMMANDS.md).

## What works

- NumPy CNN trained from random weights for 199 epochs; validation-selected checkpoint from epoch 169.
- 1,316 parameters; no pretrained model or remote AI API.
- Four synthetic visual states: READY, BUSY, DONE, OTHER.
- 99.2188% quantized accuracy on 128 generated test images; float accuracy 100.0000%.
- Integer/float class agreement on that set: 99.2188%.
- Native C++ full/head integer logits match the Python reference on 32 comparisons.
- Full activation arena is exactly 38,448 bytes, verified on the host.
- Bounded host task queue and C++ owned-buffer erasure components are included.
- GPU training source exports the same parameter format; it has not run in the user's GPU environment.

The OTHER class is an X-shaped generated fixture, not a validated unknown-screen detector. Synthetic accuracy does not establish real desktop accuracy.

## Arduino UNO Q deployment

UNO Q combines a Linux-capable processor with 4 GB system RAM and a separate STM32 microcontroller with 786 KB SRAM. See [UNO_Q.md](UNO_Q.md) for the memory split, proposed Linux/MCU roles, privacy settings, and hardware verification boundary.

The full small model can be a Linux-side component. Head-only export remains available for the MCU, which receives 16 INT8 encoder features. The four-state classifier still does not provide arbitrary language/task planning.

The persistent native classifier service keeps weights loaded, processes frames in one process, and clears owned buffers after each request. It performs no desktop actions. The Linux USB/input path and Arduino Bridge transport remain to be implemented and tested on the board.

## Start locally

Python 3.12+:

```powershell
python -m pip install -r requirements.txt
python -m local_model.demo_app
```

The synthetic demo is a local test window. It does not control the desktop.

```powershell
python -m local_model.demo_app --state READY --save build/ready.png
python -m local_model.predict --image build/ready.png
python -m local_model.predict --image build/ready.png --mode partitioned --features-out build/features.raw
```

Development image exports deliberately save requested files. They are separate from active RAM-only task processing.

## Rebuild from random initialization

```powershell
python -m local_model.dataset --out data/demo_dataset.npz --samples-per-class 128 --seed 7
python -m local_model.train --data data/demo_dataset.npz --outdir artifacts --epochs 200 --batch-size 16 --lr 0.003 --patience 30
python -m local_model.quantize --model artifacts/model.npz --data data/demo_dataset.npz --outdir artifacts
```

Architecture: 64x64 grayscale -> valid conv8/ReLU -> maxpool2 -> valid conv16/ReLU -> global average pool -> dense4 logits.

The project implements forward passes, backpropagation, Adam, quantization, and fixed-memory inference. Gradient tests check both convolutions and the dense layer. The integer runtime uses INT8 tensors, INT32 accumulation, INT64 requantization intermediates, and Q24 multipliers.

**The export is a custom format, not a LiteRT file.**

## Exports

- `artifacts/model.npz`: trained float parameters.
- `artifacts/model_int8.npz`: full integer reference parameters.
- `artifacts/model_encoder.npz`: computer-side encoder parameters.
- `artifacts/model_head.npz`: decision-head parameters.
- `model_data/model_data.h`: full C++ model.
- `model_data/model_head.h`: head-only C++ model.
- `artifacts/golden/`: generated test images, features, and expected logits.
- `artifacts/*report.json`: training, quantization, and native verification evidence.

Full raw weights/biases are 1,400 bytes plus 12 multiplier bytes. Head raw weights/biases are 80 bytes plus four multiplier bytes. These are not complete firmware sizes.

## Ampere virtual GPU training

The user confirmed Ampere. NVIDIA A30 is an Ampere device; a VM/MIG assignment may expose less than the physical device's full memory.

Install a compatible CUDA PyTorch build in the training environment using the official selector:
https://pytorch.org/get-started/locally/

Then run:

```powershell
python -m pip install -r requirements.txt
python -m local_model.train_gpu --data data/demo_dataset.npz --outdir artifacts_gpu --device cuda --epochs 60
python -m local_model.quantize --model artifacts_gpu/model.npz --data data/demo_dataset.npz --outdir artifacts_gpu --header-dir artifacts_gpu/model_data
python scripts/verify_cpp.py --artifact-dir artifacts_gpu --model-data-dir artifacts_gpu/model_data
```

The trainer starts from random weights and reports the visible GPU, training time, and peak allocated/reserved GPU memory. CUDA requests fail rather than silently using CPU.

The supplied weights were trained through the verified NumPy CPU path. No A30/VM training run is claimed.

Docker examples:

```text
docker build -t personal-assistant-model .
docker run --rm personal-assistant-model
docker build -f Dockerfile.gpu -t personal-assistant-train-gpu .
docker run --rm --gpus all personal-assistant-train-gpu --data data/demo_dataset.npz --outdir artifacts_gpu --device cuda --epochs 60
```

Mount an absolute local output directory at /app/artifacts_gpu if the training result must survive container removal. Docker engine access was unavailable here; image builds and GPU passthrough are unverified. Training containers run on the development computer or GPU VM; the native runtime container targets UNO Q Linux. Confirm whether event storage limits include build images/toolchains.

## Active task lifecycle

    RAM queue -> one active job -> recognize/execute -> result
    -> release held input -> overwrite per-task buffers -> next job

- `local_model/task_queue.py` is a bounded RAM-only FIFO.
- `firmware/task_state.cpp` clears explicit state and registered regions.
- Completion, cancellation, exception, and timeout paths must invoke cleanup.
- Pending requests stay in RAM until processed or cancelled.
- Firmware/model weights remain available; active request text, observations, features, and action history are cleared.
- No online learning from active tasks is performed.

Keeping immutable weights loaded supports speed. Fixed arenas are overwritten/reset rather than repeatedly allocated and freed.

These components cover their owned buffers only. They do not prove erasure of Python/OS/SDK copies, swap, crash dumps, registers, or physical remanence. Hardware integration must register every private observation/transport buffer and audit persistence in platform libraries.

See [PRIVACY_AND_MEMORY.md](PRIVACY_AND_MEMORY.md).

## Real data

Real model training needs labeled application sessions. The capture tool is a deliberate development feature and saves local images:

```powershell
python -m local_model.capture --bbox 100 100 356 356 --session train01 --label READY --count 20
python -m local_model.prepare_data --root data/captures --train-session train01 train02 --val-session validation01 --test-session test01 --out data/real_dataset.npz
```

Collect all four classes in training and use genuinely separate validation/test sessions. Captures/real datasets are ignored by Git and excluded from Docker context.

## General task assistant scope

The requested product accepts user-defined text, suggests completions, asks clarifying questions, plans tasks, and uses tools such as web search and calendar scheduling. The current CNN is only a small visual component for that future system.

General-language reasoning is proposed on the UNO Q Linux processor, with an optional independent computer mode. UNO Q can host local Linux models, but the supplied model does not provide broad language/planning capability. Training a capable planner from scratch also requires a suitable language/action dataset and evaluation; this synthetic classifier must not be presented as that planner.

Task suggestions and clarification context should use the current job's RAM context, not saved previous task history. Ordinary task completion must not update model weights.

Local AI does not mean every tool is offline: web searches intentionally send queries to the search service, and calendar changes are persisted by the chosen calendar service. The product should disclose those specific transfers rather than promise no data ever leaves the computer.

## Checks

Requires a g++ compiler supporting C++17 for native verification:

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
python scripts/verify_cpp.py
python scripts/verify_lifecycle.py
python scripts/verify_service.py
```

The C++ core is portable source, not a verified UNO Q firmware image. ARM64/STM32 toolchain compatibility, flash placement, stack usage, and input execution need target-specific work on the confirmed UNO Q 4 GB target.

## Source publication

Intended repository: https://github.com/HawaiianOperator/Personal-assistant

The connected GitHub integration rejected file writes with 403. No successful push is claimed. The local package contains the source and generated demonstration artifacts.

## Keyword command examples

    python -m local_model.commands "Google Nvidia"
    python -m local_model.commands "Search Nvidia"
    python -m local_model.commands "Compute 2 + 2"
    python -m local_model.commands --suggest comp

Search returns a URL by default. Add --open deliberately to request opening the query in the local browser. No browser was opened during automated verification. See COMMANDS.md for arithmetic bounds and clarification behavior.


## Capability boundary

Training from scratch remains mandatory. Unsupported commands, requests over the input limit, and unsupported calculations return status `rejected` with exactly: "I am sorry the task is too complicated, try an less complicated task". Empty requests and missing command arguments ask for clarification. Rejection clears the pending command and does not execute a partial task.

This boundary is deterministic, not a trained general difficulty estimator. Search only prepares or opens the literal query; it does not compare results or complete actions described inside the query. The trained visual classifier is separate and does not understand arbitrary tasks.

## Jupyter training environment

From the project folder, in your training VM environment:

```text
python -m pip install -r requirements-jupyter.txt
python -m jupyterlab notebooks/train_from_scratch.ipynb
```

Choose the kernel belonging to that environment. For CUDA, install a VM-compatible CUDA PyTorch build first; the notebook checks GPU visibility before training. Set BACKEND to numpy for the existing handwritten CPU trainer. Notebook training and export call the same project modules as the CLI. Outputs use artifacts_jupyter, separately from supplied weights. This notebook currently trains the visual classifier; text intent/rejection training is now in notebooks/train_text_from_scratch.ipynb; see TEXT_MODEL.md. Do not expose the Jupyter server publicly without authentication.

## Terminal text requests on UNO Q

See [TEXT_MODEL.md](TEXT_MODEL.md) and notebooks/train_text_from_scratch.ipynb for the new separate from-scratch text intent model. The minimal uno-q-text.zip runs via python3 run_assistant.py without NumPy, PyTorch, a compiler or a UI. Arithmetic returns values; search returns a link for the computer companion to open. Unsupported tasks return the configured refusal. This does not add broad planning to the visual classifier.
