"""Capture an explicitly selected desktop region into a local labeled session.

This tool has no network calls and does not move the mouse or type keys.
Capturing requires an explicit --bbox and a deliberate command invocation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import math
from pathlib import Path
import re
import secrets
import time

from PIL import ImageGrab

from .dataset import CLASS_NAMES

_SESSION_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z")


def validate_session_name(session: str) -> str:
    if not isinstance(session, str) or not _SESSION_PATTERN.fullmatch(session):
        raise ValueError("session must be 1-64 letters, digits, underscores, or hyphens, beginning with a letter or digit")
    return session


def validate_bbox(bbox: tuple[int, int, int, int] | list[int]) -> tuple[int, int, int, int]:
    if len(bbox) != 4 or any(type(value) is not int for value in bbox):
        raise ValueError("bbox must contain four integer coordinates: x0 y0 x1 y1")
    x0, y0, x1, y1 = bbox
    if x1 <= x0 or y1 <= y0:
        raise ValueError("bbox must have positive width and height")
    # Negative origins can represent a monitor left/above the primary display.
    if (x1 - x0) * (y1 - y0) > 16_777_216:
        raise ValueError("bbox exceeds the 16,777,216-pixel capture limit; select a smaller region")
    return x0, y0, x1, y1


def _capture_directory(root: Path, session: str, label: str) -> Path:
    validate_session_name(session)
    if label not in CLASS_NAMES:
        raise ValueError(f"label must be one of {CLASS_NAMES}")
    root = root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    directory = (root / session / label).resolve()
    if not directory.is_relative_to(root):
        raise ValueError("session directory resolves outside the capture root")
    directory.mkdir(parents=True, exist_ok=True)
    # Check again in case a parent was a link to an external path.
    if not directory.resolve().is_relative_to(root):
        raise ValueError("session directory resolves outside the capture root")
    return directory


def capture_samples(
    *,
    bbox: tuple[int, int, int, int],
    session: str,
    label: str,
    out: Path,
    count: int = 1,
    interval: float = 1.0,
    countdown: int = 3,
) -> list[Path]:
    bbox = validate_bbox(bbox)
    if type(count) is not int or not 1 <= count <= 10_000:
        raise ValueError("count must be an integer from 1 to 10000")
    if not math.isfinite(interval) or interval <= 0:
        raise ValueError("interval must be a positive finite number")
    if type(countdown) is not int or not 0 <= countdown <= 60:
        raise ValueError("countdown must be an integer from 0 to 60")
    directory = _capture_directory(out, session, label)
    print(f"LOCAL CAPTURE: region={bbox}; session={session}; label={label}; images={count}", flush=True)
    print(f"PNG files will be saved only in {directory}. Ctrl+C stops capture.", flush=True)
    for remaining in range(countdown, 0, -1):
        print(f"Starting in {remaining}...", flush=True)
        time.sleep(1)
    paths = []
    for index in range(count):
        image = ImageGrab.grab(bbox=bbox, all_screens=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        path = directory / f"frame_{timestamp}_{secrets.token_hex(4)}.png"
        with path.open("xb") as destination:
            image.save(destination, format="PNG")
        paths.append(path)
        print(f"Saved {index + 1}/{count}: {path.name}", flush=True)
        if index + 1 < count:
            time.sleep(interval)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bbox", type=int, nargs=4, required=True, metavar=("X0", "Y0", "X1", "Y1"))
    parser.add_argument("--session", required=True, help="Capture-session identifier; keep related frames in the same session.")
    parser.add_argument("--label", choices=CLASS_NAMES, required=True)
    parser.add_argument("--out", type=Path, default=Path("data/captures"))
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--countdown", type=int, default=3)
    args = parser.parse_args()
    try:
        capture_samples(
            bbox=tuple(args.bbox),
            session=args.session,
            label=args.label,
            out=args.out,
            count=args.count,
            interval=args.interval,
            countdown=args.countdown,
        )
    except KeyboardInterrupt:
        print("Capture stopped. Previously saved local files were kept.", flush=True)
        raise SystemExit(130) from None
    except (ValueError, OSError) as error:
        parser.exit(1, f"Capture failed: {error}\n")


if __name__ == "__main__":
    main()
