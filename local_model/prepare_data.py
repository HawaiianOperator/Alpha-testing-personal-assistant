"""Prepare local captured PNG sessions with explicit, disjoint data splits.

No random frame splitting is performed. Supply all three session lists so
related frames cannot accidentally be distributed between train and test.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import numpy as np
from PIL import Image, UnidentifiedImageError

from .dataset import CLASS_NAMES

_SESSION_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z")


def _session_path(root: Path, name: str) -> Path:
    if not isinstance(name, str) or not _SESSION_PATTERN.fullmatch(name):
        raise ValueError(f"Invalid session identifier: {name!r}")
    path = (root / name).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Session {name!r} resolves outside the capture root")
    if not path.is_dir():
        raise ValueError(f"Session directory does not exist: {name}")
    return path


def _validate_splits(root: Path, session_lists: dict[str, list[str]]) -> dict[str, list[tuple[str, Path]]]:
    paths: dict[str, list[tuple[str, Path]]] = {}
    seen_names: set[str] = set()
    seen_paths: set[Path] = set()
    for split in ("train", "val", "test"):
        names = session_lists[split]
        if not names:
            raise ValueError(f"{split} must explicitly contain at least one session")
        entries = []
        for name in names:
            if name in seen_names:
                raise ValueError(f"Session {name!r} appears more than once; all splits must be disjoint")
            path = _session_path(root, name)
            if path in seen_paths:
                raise ValueError(f"Session {name!r} aliases an already selected directory")
            seen_names.add(name)
            seen_paths.add(path)
            entries.append((name, path))
        paths[split] = entries
    return paths


def prepare_dataset(
    root: Path,
    train_sessions: list[str],
    val_sessions: list[str],
    test_sessions: list[str],
    *,
    max_samples: int = 10_000,
) -> dict[str, np.ndarray]:
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Capture root does not exist: {root}")
    if type(max_samples) is not int or max_samples <= 0:
        raise ValueError("max_samples must be a positive integer")
    session_lists = {"train": train_sessions, "val": val_sessions, "test": test_sessions}
    session_paths = _validate_splits(root, session_lists)
    records: dict[str, list[tuple[str, int, Path]]] = {}
    total = 0
    for split, sessions in session_paths.items():
        entries = []
        for session_name, session_root in sessions:
            for label_index, label in enumerate(CLASS_NAMES):
                label_root = session_root / label
                if not label_root.exists():
                    continue
                resolved_label = label_root.resolve()
                if not resolved_label.is_relative_to(root) or not resolved_label.is_dir():
                    raise ValueError(f"Invalid label directory: {session_name}/{label}")
                for candidate in sorted(label_root.iterdir()):
                    if candidate.suffix.lower() != ".png":
                        continue
                    path = candidate.resolve()
                    if not path.is_relative_to(root):
                        raise ValueError(f"Image resolves outside the capture root: {candidate}")
                    if path.is_file():
                        entries.append((session_name, label_index, path))
                        total += 1
                        if total > max_samples:
                            raise ValueError(f"Dataset exceeds the explicit {max_samples}-sample preparation limit")
        if not entries:
            raise ValueError(f"{split} has no PNG samples in the expected session/CLASS folders")
        records[split] = entries

    training_labels = {label for _, label, _ in records["train"]}
    missing = [CLASS_NAMES[index] for index in range(len(CLASS_NAMES)) if index not in training_labels]
    if missing:
        raise ValueError("Training data is missing classes: " + ", ".join(missing))

    result: dict[str, np.ndarray] = {}
    counts: dict[str, dict[str, int]] = {}
    for split, entries in records.items():
        x = np.empty((len(entries), 1, 64, 64), dtype=np.float32)
        y = np.empty(len(entries), dtype=np.int64)
        sessions = []
        sources = []
        for index, (session, label, path) in enumerate(entries):
            try:
                with Image.open(path) as image:
                    if image.format != "PNG":
                        raise ValueError(f"File does not contain a PNG image: {path}")
                    gray = image.convert("L").resize((64, 64), Image.Resampling.BILINEAR)
                    x[index, 0] = np.asarray(gray, dtype=np.float32) / np.float32(255.0)
            except (OSError, UnidentifiedImageError) as error:
                raise ValueError(f"Could not read PNG image {path}: {error}") from error
            y[index] = label
            sessions.append(session)
            sources.append(path.relative_to(root).as_posix())
        result[f"x_{split}"] = x
        result[f"y_{split}"] = y
        result[f"session_{split}"] = np.asarray(sessions)
        result[f"source_{split}"] = np.asarray(sources)
        counts[split] = {
            label: int(np.count_nonzero(y == index))
            for index, label in enumerate(CLASS_NAMES)
        }

    metadata = {
        "synthetic": False,
        "provenance": "local_labeled_screen_captures",
        "real_desktop_data": True,
        "class_names": list(CLASS_NAMES),
        "split_method": "explicit_disjoint_capture_sessions",
        "split_sessions": {name: list(values) for name, values in session_lists.items()},
        "split_class_counts": counts,
        "input_shape": [1, 64, 64],
        "preprocessing": {"mode": "L", "resize": [64, 64], "resampling": "bilinear", "range": [0.0, 1.0]},
        "samples": total,
        "limitations": [
            "Session separation does not itself prove broad application or OS generalization.",
            "Label correctness and non-duplicate capture sessions must be reviewed by the dataset owner.",
        ],
    }
    encoded = np.asarray(json.dumps(metadata, sort_keys=True))
    result["metadata_json"] = encoded
    result["metadata"] = encoded
    result["class_names"] = np.asarray(CLASS_NAMES)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/captures"))
    parser.add_argument("--out", type=Path, default=Path("data/real_dataset.npz"))
    parser.add_argument("--train-session", nargs="+", required=True)
    parser.add_argument("--val-session", nargs="+", required=True)
    parser.add_argument("--test-session", nargs="+", required=True)
    parser.add_argument("--max-samples", type=int, default=10_000)
    args = parser.parse_args()
    try:
        data = prepare_dataset(
            args.root,
            args.train_session,
            args.val_session,
            args.test_session,
            max_samples=args.max_samples,
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.out, **data)
    except (ValueError, OSError) as error:
        parser.exit(1, f"Dataset preparation failed: {error}\n")
    metadata = json.loads(str(data["metadata_json"]))
    print(json.dumps({
        "dataset": str(args.out),
        "synthetic": False,
        "split_samples": {split: len(data[f"y_{split}"]) for split in ("train", "val", "test")},
        "split_class_counts": metadata["split_class_counts"],
    }))


if __name__ == "__main__":
    main()
