# Text model trained from scratch

The terminal runtime accepts text requests using a new learned intent classifier, separate from the image model. No pretrained weights, language-model API, NumPy, PyTorch or network are needed during inference. Training uses NumPy and starts from random weights every time.

## Run on UNO Q Linux

Extract uno-q-text.zip, then run:

```sh
cd uno-q-text
python3 run_assistant.py
```

Type Compute 2 + 2, Calculate 17 + 29, Search Nvidia, or Generate an image of the Empire State Building. Type exit to quit. Compute or Search alone asks for the missing argument. cancel clears that pending label. The exact rejection is: I am sorry the task is too complicated, try an less complicated task

Search returns a Google URL; the offline board does not open a computer browser or retrieve/search web pages. Your partner can consume one JSON reply per line using python3 run_assistant.py --jsonl. This is a local process protocol, not a network or USB service. Returned URLs must be opened by a separately implemented computer companion.

## Model and boundaries

A 6,147-parameter softmax classifier learns three intents: search, compute and reject. Stable hashed word/character features have 2,048 dimensions. Text classification is learned; argument extraction, bounded arithmetic execution and multi-step refusal are explicit code. Fixed score/margin thresholds refuse uncertain inputs; scores are not calibrated certainty. Only supported phrasing can execute. It is not a general language model or a universal task-complexity detector. There is no image/text generation or calendar scheduling.

The bundled model was trained from random weights locally on CPU using 682 synthetic labeled examples and evaluated on 18 separate fixed prompts. These are a small same-family fixture check, not evidence of arbitrary language accuracy. Runtime tests also check unsupported operations and clarification. Add diverse independently labeled data and new held-out evaluations before broadening capabilities. Intent examples contain only synthetic development prompts.

## Define and retrain in Jupyter

Open notebooks/train_text_from_scratch.ipynb. Edit data/intent_training.jsonl (each line has text and label, with labels search/compute/reject). The generator can replace that file; skip generation when preserving edited examples. Existing training refuses exact overlap with its fixed evaluation prompts. Semantic/template overlap is still possible and requires independently collected evaluation data. Model files and training reports intentionally persist; active requests are not added to training.

```sh
python -m local_model.train_intent --data data/intent_training.jsonl --outdir artifacts_intent --epochs 400
python scripts/package_text.py
```

An A30 is not required for this small classifier. This CPU NumPy trainer can run inside the existing GPU Jupyter environment. The current deployable text model has not been executed on UNO Q here; its standard-library runtime passes host checks without third-party site packages. No board latency is claimed.

## Privacy

The runtime creates no task log/history files and performs no network requests or online learning. After completion/rejection it retains no task context; clarification stores only a command label. Python strings/OS/terminal copies cannot be guaranteed securely erased. Terminal scrollback may display requests and answers. Immutable weights are saved on the board. Shell, logging, swap and crash dump configuration remain outside this application's guarantees.
