"""Train the custom NumPy CNN locally from random initialization."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

# Small matrices are usually faster without a large BLAS thread pool.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

try:
    from .network import Adam, CLASS_NAMES, Network, cross_entropy
except ImportError:
    from network import Adam, CLASS_NAMES, Network, cross_entropy


def load_dataset(path):
    with np.load(path, allow_pickle=False) as archive:
        result = {}
        for split in ("train", "val", "test"):
            x, y = archive[f"x_{split}"], archive[f"y_{split}"]
            x = Network._validate_input(x)
            if y.ndim != 1 or len(y) != len(x) or y.dtype.kind not in "iu":
                raise ValueError(f"Invalid {split} labels.")
            if np.any(y < 0) or np.any(y >= 4):
                raise ValueError(f"{split} labels must be in [0, 3].")
            result[f"x_{split}"] = x.copy()
            result[f"y_{split}"] = y.astype(np.int64)
        metadata = {}
        for key in ("metadata_json", "metadata"):
            if key in archive:
                raw = archive[key].item()
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8")
                if isinstance(raw, str):
                    metadata = json.loads(raw)
                    break
        if "synthetic" in archive:
            metadata["synthetic"] = bool(archive["synthetic"].item())
        result["metadata"] = metadata
    if not isinstance(metadata, dict):
        raise ValueError("Dataset metadata must be a JSON object.")
    return result


def evaluate(model, x, y, batch_size=32):
    total_loss, logits_list = 0.0, []
    for start in range(0, len(x), batch_size):
        bx, by = x[start:start + batch_size], y[start:start + batch_size]
        logits = model.forward(bx)
        loss, _ = cross_entropy(logits, by)
        total_loss += loss * len(bx)
        logits_list.append(logits)
    logits = np.concatenate(logits_list)
    predicted = logits.argmax(axis=1)
    confusion = np.zeros((4, 4), dtype=np.int64)
    for actual, prediction in zip(y, predicted):
        confusion[int(actual), int(prediction)] += 1
    per_class = []
    for index in range(4):
        count = int(confusion[index].sum())
        per_class.append(float(confusion[index, index] / count) if count else None)
    return {
        "loss": float(total_loss / len(x)),
        "accuracy": float(np.mean(predicted == y)),
        "count": int(len(y)),
        "confusion_matrix": confusion.tolist(),
        "per_class_recall": per_class,
    }


def train(args):
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0 or args.patience < 1:
        raise ValueError("Epochs, batch size, learning rate, and patience must be positive.")
    dataset_path = Path(args.data)
    data = load_dataset(dataset_path)
    output = Path(args.outdir)
    output.mkdir(parents=True, exist_ok=True)
    model = Network(seed=args.seed)
    optimizer = Adam(model.params, learning_rate=args.learning_rate)
    rng = np.random.default_rng(args.seed)
    best_loss, best_epoch, stale_epochs = float("inf"), 0, 0
    best_state = model.state_dict()
    history = []
    started = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        order = rng.permutation(len(data["x_train"]))
        training_loss = 0.0
        for start in range(0, len(order), args.batch_size):
            indices = order[start:start + args.batch_size]
            loss, gradients = model.loss_and_gradients(
                data["x_train"][indices], data["y_train"][indices]
            )
            optimizer.step(model.params, gradients)
            training_loss += loss * len(indices)
        validation = evaluate(model, data["x_val"], data["y_val"], args.batch_size)
        row = {
            "epoch": epoch,
            "train_loss": float(training_loss / len(order)),
            "val_loss": validation["loss"],
            "val_accuracy": validation["accuracy"],
        }
        history.append(row)
        if validation["loss"] < best_loss - 1e-7:
            best_loss, best_epoch, stale_epochs = validation["loss"], epoch, 0
            best_state = model.state_dict()
            model.save(output / "model.npz")
        else:
            stale_epochs += 1
        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            print(f"epoch {epoch:03d} train_loss={row['train_loss']:.4f} "
                  f"val_loss={row['val_loss']:.4f} val_acc={row['val_accuracy']:.3f}",
                  flush=True)
        if stale_epochs >= args.patience:
            print(f"Early stop after {epoch} epochs; best validation epoch {best_epoch}.", flush=True)
            break
    model.load_state_dict(best_state)
    model.save(output / "model.npz")
    metadata = data["metadata"]
    class_names = metadata.get("class_names", list(CLASS_NAMES))
    report = {
        "architecture": "conv8_pool_conv16_gap_dense4",
        "input_shape": [1, 64, 64],
        "initialization": "random; no pretrained weights",
        "seed": args.seed,
        "optimizer": "Adam",
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "epochs_requested": args.epochs,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "selection_rule": "minimum validation cross-entropy",
        "parameter_count": model.parameter_count,
        "float32_parameter_bytes": sum(value.nbytes for value in model.params.values()),
        "class_names": class_names,
        "dataset": {
            "file": dataset_path.name,
            "sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
            "synthetic": metadata.get("synthetic", "unknown"),
            "metadata": metadata,
            "split_counts": {split: int(len(data[f"y_{split}"])) for split in ("train", "val", "test")},
        },
        "validation": evaluate(model, data["x_val"], data["y_val"], args.batch_size),
        "test": evaluate(model, data["x_test"], data["y_test"], args.batch_size),
        "training_seconds": round(time.monotonic() - started, 3),
        "history": history,
        "limitations": [
            "Performance applies only to the supplied dataset and its declared split.",
            "Synthetic test performance does not establish accuracy on real desktop screenshots.",
            "No Arduino UNO Q inference latency, total runtime memory, Bridge transport, or HID loop is measured here.",
        ],
    }
    report["model_sha256"] = hashlib.sha256((output / "model.npz").read_bytes()).hexdigest()
    (output / "training_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"Saved random-initialized training result to {output}; "
          f"test_acc={report['test']['accuracy']:.3f}, parameters={model.parameter_count}.", flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Local NPZ with train/val/test arrays.")
    parser.add_argument("--outdir", default="artifacts")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--learning-rate", "--lr", type=float, default=0.003)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--log-every", type=int, default=5)
    args = parser.parse_args(argv)
    if args.log_every < 1:
        parser.error("--log-every must be positive.")
    train(args)


if __name__ == "__main__":
    main()
