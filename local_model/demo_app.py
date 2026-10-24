"""Display the synthetic UI fixture locally; this app never controls the desktop."""
from __future__ import annotations

import argparse
from pathlib import Path

from .dataset import CLASS_NAMES, make_contact_sheet, render_state


def run_app(initial_state: str = "READY", seed: int = 7) -> None:
    import tkinter as tk
    from tkinter import ttk
    from PIL import ImageTk
    import numpy as np
    from .quantize import load_quantized, predict_quantized
    model_path = Path(__file__).resolve().parent.parent / "artifacts/model_int8.npz"
    model_params = load_quantized(model_path) if model_path.exists() else None

    root = tk.Tk()
    root.title("Personal Assistant - Synthetic UI Fixture")
    root.resizable(False, False)
    frame = ttk.Frame(root, padding=16)
    frame.grid()
    ttk.Label(frame, text="Synthetic model demo", font=("Segoe UI", 15, "bold")).grid(row=0, column=0, columnspan=4)
    ttk.Label(frame, text="Local fixture only - no desktop control or network calls.").grid(row=1, column=0, columnspan=4, pady=(4, 10))
    panel = ttk.Label(frame)
    panel.grid(row=2, column=0, columnspan=4)
    status = tk.StringVar()
    ttk.Label(frame, textvariable=status).grid(row=3, column=0, columnspan=4, pady=10)
    ttk.Label(frame, text="Capture only the image panel above. State labels are outside the model crop.").grid(row=4, column=0, columnspan=4)
    active = {"state": initial_state.upper(), "theme": seed, "pending": None}

    def show_state(name: str) -> None:
        if active["pending"] is not None:
            root.after_cancel(active["pending"])
            active["pending"] = None
        active["state"] = name
        image = render_state(name, seed, size=256, session_seed=active["theme"])
        photo = ImageTk.PhotoImage(image)
        panel.configure(image=photo)
        panel.image = photo
        if model_params is not None:
            model_image = render_state(name, seed, size=64, session_seed=active["theme"])
            pixels = np.asarray(model_image, dtype=np.uint8).reshape(1, 1, 64, 64)
            prediction = int(predict_quantized(model_params, pixels)[0].argmax())
            status.set(f"Ground truth: {name} | Model: {CLASS_NAMES[prediction]} | Enter: start")
        else:
            status.set(f"Ground truth: {name} | Model not exported yet")

    def start(event=None) -> None:
        if active["state"] == "READY":
            show_state("BUSY")
            active["pending"] = root.after(1800, lambda: show_state("DONE"))

    def next_theme() -> None:
        active["theme"] += 7919
        show_state(active["state"])

    for column, name in enumerate(CLASS_NAMES):
        ttk.Button(frame, text=name, command=lambda chosen=name: show_state(chosen)).grid(row=5, column=column, padx=3, pady=(12, 6))
    ttk.Button(frame, text="Change theme", command=next_theme).grid(row=6, column=0, columnspan=4)
    root.bind("<Return>", start)
    root.bind("<Escape>", lambda event: show_state("READY"))
    show_state(active["state"])
    root.mainloop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", choices=CLASS_NAMES, default="READY")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--save", type=Path, help="Save one 64x64 input image without opening a window.")
    parser.add_argument("--contact-sheet", type=Path, help="Save a labeled synthetic preview without opening a window.")
    args = parser.parse_args()
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        render_state(args.state, args.seed).save(args.save)
    if args.contact_sheet:
        args.contact_sheet.parent.mkdir(parents=True, exist_ok=True)
        make_contact_sheet(args.seed).save(args.contact_sheet)
    if not args.save and not args.contact_sheet:
        run_app(args.state, args.seed)


if __name__ == "__main__":
    main()
