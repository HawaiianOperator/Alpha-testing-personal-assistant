"""Generate a controlled, synthetic UI-state dataset locally.

These images are a demonstration fixture, not captured desktop data or evidence
of accuracy on real applications. The four classes share palette distributions.
Train/validation/test sessions are generated independently.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

CLASS_NAMES = ("READY", "BUSY", "DONE", "OTHER")
_IMAGE_SIZE = 64


def _state_index(state: int | str) -> int:
    if isinstance(state, str):
        try:
            return CLASS_NAMES.index(state.strip().upper())
        except ValueError:
            raise ValueError(f"state must be one of {CLASS_NAMES}") from None
    if isinstance(state, (int, np.integer)) and not isinstance(state, bool) and 0 <= int(state) < len(CLASS_NAMES):
        return int(state)
    raise ValueError(f"state must be an index from 0 to {len(CLASS_NAMES) - 1} or a class name")


def render_state(
    state: int | str,
    seed: int,
    size: int = 64,
    *,
    session_seed: int | None = None,
) -> Image.Image:
    """Render one grayscale icon panel; class name text is never in the crop.

    A session controls its palette and window style. Per-image seed controls
    positioning, scale, and small pixel noise. Using the same seed/session for
    every state produces the same background and palette.
    """
    label = _state_index(state)
    if not isinstance(size, int) or isinstance(size, bool) or not 16 <= size <= 1024:
        raise ValueError("size must be an integer between 16 and 1024")
    rng = np.random.default_rng(seed)
    theme = np.random.default_rng(seed if session_seed is None else session_seed)

    # Theme samples are independent of label. Keep adequate contrast while
    # varying brightness, border geometry, icon position, and polarity.
    dark = bool(theme.integers(0, 2))
    if dark:
        background = int(theme.integers(20, 65))
        panel = int(theme.integers(65, 100))
        foreground = int(theme.integers(205, 251))
        chrome = min(255, background + 20)
    else:
        background = int(theme.integers(195, 236))
        panel = int(theme.integers(150, 185))
        foreground = int(theme.integers(5, 45))
        chrome = max(0, background - 20)
    border_radius = int(theme.integers(2, 6))
    image = Image.new("L", (_IMAGE_SIZE, _IMAGE_SIZE), background)
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((3, 3, 60, 60), radius=border_radius, fill=panel, outline=foreground, width=1)
    draw.rectangle((5, 5, 58, 12), fill=chrome)
    for x in (8, 13, 18):
        draw.ellipse((x, 7, x + 2, 9), fill=foreground)

    cx = 32 + int(rng.integers(-3, 4))
    cy = 36 + int(rng.integers(-2, 3))
    radius = int(rng.integers(12, 17))
    stroke = int(rng.integers(4, 7))
    if label == 0:  # READY: filled play triangle.
        draw.polygon(
            [(cx - radius + 3, cy - radius), (cx - radius + 3, cy + radius), (cx + radius, cy)],
            fill=foreground,
        )
    elif label == 1:  # BUSY: three separated progress bars.
        bar_height = max(3, stroke - 1)
        for y in (cy - 11, cy - 1, cy + 9):
            draw.rectangle((cx - radius, y, cx + radius, y + bar_height), fill=foreground)
    elif label == 2:  # DONE: a check mark.
        draw.line(
            [(cx - radius, cy), (cx - radius // 3, cy + radius // 2), (cx + radius, cy - radius // 2)],
            fill=foreground,
            width=stroke,
            joint="curve",
        )
    else:  # OTHER: an unsupported-dialog X icon.
        draw.line((cx - radius, cy - radius, cx + radius, cy + radius), fill=foreground, width=stroke)
        draw.line((cx - radius, cy + radius, cx + radius, cy - radius), fill=foreground, width=stroke)

    # Noise is label-independent, deterministic, and small enough to preserve
    # state identity. It prevents identical-image leakage across sessions.
    pixels = np.asarray(image, dtype=np.int16)
    noise = rng.integers(-2, 3, size=(_IMAGE_SIZE, _IMAGE_SIZE), dtype=np.int16)
    pixels = np.clip(pixels + noise, 0, 255).astype(np.uint8)
    image = Image.fromarray(pixels)
    if size != _IMAGE_SIZE:
        image = image.resize((size, size), Image.Resampling.NEAREST)
    return image


def build_dataset(samples_per_class: int = 128, seed: int = 7) -> dict[str, np.ndarray]:
    """Create balanced splits with disjoint theme/capture-session identifiers.

    samples_per_class controls each training class. Validation and test each
    use max(4, ceil(training_count / 4)) examples per class.
    """
    if not isinstance(samples_per_class, int) or isinstance(samples_per_class, bool) or samples_per_class < 4:
        raise ValueError("samples_per_class must be an integer of at least 4")
    rng = np.random.default_rng(seed)
    count_small = max(4, (samples_per_class + 3) // 4)
    split_counts = {"train": samples_per_class, "val": count_small, "test": count_small}
    result: dict[str, np.ndarray] = {}
    sessions: dict[str, list[int]] = {}
    used_sessions: set[int] = set()
    for split in ("train", "val", "test"):
        count = split_counts[split]
        session_count = min(8 if split == "train" else 3, count)
        split_sessions: list[int] = []
        while len(split_sessions) < session_count:
            session = int(rng.integers(0, np.iinfo(np.int32).max))
            if session not in used_sessions:
                split_sessions.append(session)
                used_sessions.add(session)
        sessions[split] = split_sessions
        n = count * len(CLASS_NAMES)
        images = np.empty((n, 1, _IMAGE_SIZE, _IMAGE_SIZE), dtype=np.float32)
        labels = np.empty(n, dtype=np.int64)
        session_ids = np.empty(n, dtype=np.int64)
        index = 0
        for label in range(len(CLASS_NAMES)):
            for sample in range(count):
                session = split_sessions[sample % session_count]
                image_seed = int(rng.integers(0, np.iinfo(np.int32).max))
                image = render_state(label, image_seed, session_seed=session)
                images[index, 0] = np.asarray(image, dtype=np.float32) / np.float32(255.0)
                labels[index] = label
                session_ids[index] = session
                index += 1
        order = rng.permutation(n)
        result[f"x_{split}"] = images[order]
        result[f"y_{split}"] = labels[order]
        result[f"session_{split}"] = session_ids[order]

    metadata = {
        "synthetic": True,
        "provenance": "synthetic_ui_generated_locally",
        "real_desktop_data": False,
        "intended_use": "controlled_demo_only",
        "class_names": list(CLASS_NAMES),
        "input_shape": [1, _IMAGE_SIZE, _IMAGE_SIZE],
        "input_range": [0.0, 1.0],
        "seed": seed,
        "training_samples_per_class": samples_per_class,
        "split_counts_per_class": split_counts,
        "split_method": "disjoint_synthetic_capture_sessions",
        "split_sessions": sessions,
        "limitations": [
            "No claim of accuracy on real desktop screens.",
            "OTHER is a synthetic X-icon class, not comprehensive unknown-screen detection.",
            "Evaluate real target application captures before enabling desktop actions.",
        ],
    }
    encoded = np.asarray(json.dumps(metadata, sort_keys=True))
    result["metadata_json"] = encoded
    result["metadata"] = encoded
    result["class_names"] = np.asarray(CLASS_NAMES)
    return result


def make_contact_sheet(seed: int = 7, samples_per_class: int = 4, cell_size: int = 96) -> Image.Image:
    """Create a labeled preview; labels are outside each model input image."""
    if not 1 <= samples_per_class <= 20:
        raise ValueError("samples_per_class must be from 1 through 20")
    width = cell_size * len(CLASS_NAMES)
    height = 42 + samples_per_class * cell_size
    sheet = Image.new("RGB", (width, height), "#f3f4f6")
    draw = ImageDraw.Draw(sheet)
    draw.text((10, 5), "SYNTHETIC UI DATA - DEMONSTRATION ONLY", fill="#111827")
    for column, name in enumerate(CLASS_NAMES):
        draw.text((column * cell_size + 10, 25), name, fill="#111827")
        for row in range(samples_per_class):
            # Same seed/session across columns isolates visual state from theme.
            image = render_state(name, seed + row * 1009, session_seed=seed + row * 7919)
            x = column * cell_size + (cell_size - 64) // 2
            y = 42 + row * cell_size + (cell_size - 64) // 2
            sheet.paste(image.convert("RGB"), (x, y))
    return sheet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/demo_dataset.npz"))
    parser.add_argument("--samples-per-class", type=int, default=128)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--preview", type=Path, help="Optional synthetic contact-sheet PNG.")
    args = parser.parse_args()
    dataset = build_dataset(args.samples_per_class, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **dataset)
    if args.preview:
        args.preview.parent.mkdir(parents=True, exist_ok=True)
        make_contact_sheet(args.seed).save(args.preview)
    sizes = {split: int(dataset[f"y_{split}"].size) for split in ("train", "val", "test")}
    print(json.dumps({"dataset": str(args.out), "synthetic": True, "split_samples": sizes}))


if __name__ == "__main__":
    main()
