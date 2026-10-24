"""Run the visual-state prototype locally without performing desktop actions."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from .quantize import load_quantized, encode_quantized, head_quantized, predict_quantized


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="artifacts/model_int8.npz")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--image", type=Path)
    inputs.add_argument("--raw", type=Path)
    parser.add_argument("--mode", choices=("full", "partitioned"), default="full")
    parser.add_argument("--features-out", type=Path)
    args = parser.parse_args()
    if args.raw:
        raw = args.raw.read_bytes()
        if len(raw) != 4096:
            parser.error("Raw input must have exactly 4096 bytes.")
        pixels = np.frombuffer(raw, dtype=np.uint8).reshape(1, 1, 64, 64)
    else:
        with Image.open(args.image) as source:
            pixels = np.asarray(source.convert("L").resize((64, 64), Image.Resampling.BILINEAR),
                                dtype=np.uint8).reshape(1, 1, 64, 64)
    params = load_quantized(args.model)
    if args.mode == "partitioned" or args.features_out:
        features = encode_quantized(params, pixels)
        logits = head_quantized(params, features)
        if args.features_out:
            args.features_out.parent.mkdir(parents=True, exist_ok=True)
            features[0].tofile(args.features_out)
    else:
        logits = predict_quantized(params, pixels)
    names = ["READY", "BUSY", "DONE", "OTHER"]
    index = int(logits[0].argmax())
    print(json.dumps({"class_id": index, "state": names[index],
                      "integer_logits": logits[0].astype(int).tolist(), "mode": args.mode,
                      "scope": "Synthetic screen-state prototype; no task intent or desktop action."}))


if __name__ == "__main__":
    main()
