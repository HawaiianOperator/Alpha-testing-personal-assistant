"""Train the same custom architecture from random weights using PyTorch/CUDA.

Use in the user's Ampere virtual environment. This module never downloads models.
The NPZ export stays compatible with the NumPy quantizer and C++ runtime.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from .train import load_dataset


def build_model(seed=7):
    import torch
    from torch import nn
    torch.manual_seed(seed)
    model = nn.Sequential(
        nn.Conv2d(1, 8, 3), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(8, 16, 3), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(16, 4),
    )
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            nn.init.zeros_(module.bias)
    return model


def export_numpy(model):
    return {
        "conv1_w": model[0].weight.detach().cpu().numpy().copy(),
        "conv1_b": model[0].bias.detach().cpu().numpy().copy(),
        "conv2_w": model[3].weight.detach().cpu().numpy().copy(),
        "conv2_b": model[3].bias.detach().cpu().numpy().copy(),
        "dense_w": model[7].weight.detach().cpu().numpy().copy(),
        "dense_b": model[7].bias.detach().cpu().numpy().copy(),
    }


def train(args):
    import torch
    from torch import nn
    torch.set_num_threads(args.cpu_threads)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no usable GPU is visible. Check VM GPU passthrough and the driver.")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats(device)
        torch.backends.cuda.matmul.allow_tf32 = not args.deterministic
        torch.backends.cudnn.allow_tf32 = not args.deterministic
        torch.backends.cudnn.benchmark = not args.deterministic
    torch.use_deterministic_algorithms(args.deterministic)
    model = build_model(args.seed).to(device)
    data = load_dataset(args.data)
    tensors = {
        key: torch.from_numpy(data[key]).to(device)
        for key in ("x_train", "y_train", "x_val", "y_val", "x_test", "y_test")
    }
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    best_loss = float("inf")
    best_state = None
    history = []
    started = time.monotonic()

    def evaluate(split):
        model.eval()
        with torch.no_grad():
            logits = model(tensors["x_" + split])
            labels = tensors["y_" + split]
            loss = float(criterion(logits, labels).item())
            predictions = logits.argmax(1)
            accuracy = float((predictions == labels).float().mean().item())
        return loss, accuracy

    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(len(tensors["y_train"]), device=device)
        total = 0.0
        for offset in range(0, len(order), args.batch_size):
            indexes = order[offset:offset + args.batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(tensors["x_train"][indexes]), tensors["y_train"][indexes])
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss.")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += float(loss.detach().item()) * len(indexes)
        val_loss, val_accuracy = evaluate("val")
        history.append({"epoch": epoch, "train_loss": total / len(order),
                        "val_loss": val_loss, "val_accuracy": val_accuracy})
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            print(f"epoch {epoch:03d} train_loss={total / len(order):.4f} "
                  f"val_loss={val_loss:.4f} val_acc={val_accuracy:.3f}", flush=True)
    model.load_state_dict(best_state)
    output = Path(args.outdir)
    output.mkdir(parents=True, exist_ok=True)
    np.savez(output / "model.npz", **export_numpy(model))
    test_loss, test_accuracy = evaluate("test")
    gpu = None
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        gpu = {"name": props.name, "total_memory": props.total_memory,
               "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
               "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)}
    report = {
        "backend": "PyTorch", "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda, "device": str(device), "gpu": gpu,
        "initialization": "random; no pretrained weights",
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "dataset_synthetic": data["metadata"].get("synthetic", "unknown"),
        "dataset_sha256": hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
        "deterministic": args.deterministic,
        "speed_settings": "fixed shapes; data preloaded to selected device; TF32 and cuDNN benchmark when CUDA/non-deterministic",
        "test": {"loss": test_loss, "accuracy": test_accuracy},
        "training_seconds": time.monotonic() - started, "history": history,
        "limitations": ["Dataset performance does not establish arbitrary desktop control.",
                        "GPU training does not establish Arduino UNO Q latency or Linux/STM32 runtime memory use."],
    }
    (output / "gpu_training_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Saved training export to {output}; test_acc={test_accuracy:.3f}.", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--outdir", default="artifacts_gpu")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.cpu_threads) < 1 or args.lr <= 0:
        parser.error("Epochs, batch size, threads, and learning rate must be positive.")
    train(args)


if __name__ == "__main__":
    main()
