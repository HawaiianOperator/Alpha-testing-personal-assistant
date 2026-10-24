"""Calibrate and export a small INT8 model with explicit Q24 arithmetic.

This exporter targets the bundled fixed-memory C++ runtime, not LiteRT. It uses
per-tensor symmetric weights, nonnegative ReLU tensors, and INT32 biases.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .network import Network

Q_SHIFT = 24
INPUT_SCALE = 1.0 / 127.0


def round_away(x):
    x = np.asarray(x)
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


def requantize(acc, multiplier, relu=False):
    product = np.asarray(acc, dtype=np.int64) * int(multiplier)
    rounded = np.sign(product) * ((np.abs(product) + (1 << 23)) >> Q_SHIFT)
    return np.clip(rounded, 0 if relu else -127, 127).astype(np.int8)


def quantize_input(pixels):
    pixels = np.asarray(pixels)
    if pixels.dtype != np.uint8 or pixels.ndim != 4 or pixels.shape[1:] != (1, 64, 64):
        raise ValueError("Expected uint8 NCHW images with shape (N,1,64,64).")
    return ((pixels.astype(np.int32) * 127 + 127) // 255).astype(np.int8)


def _conv(x, weight, bias):
    n, channels, height, width = x.shape
    windows = np.lib.stride_tricks.sliding_window_view(x, (3, 3), axis=(2, 3))
    columns = np.ascontiguousarray(windows.transpose(0, 2, 3, 1, 4, 5)).reshape(-1, channels * 9)
    acc = columns.astype(np.int32) @ weight.reshape(weight.shape[0], -1).astype(np.int32).T
    acc += bias
    return acc.reshape(n, height - 2, width - 2, weight.shape[0]).transpose(0, 3, 1, 2)


def encode_quantized(params, pixels):
    x = quantize_input(pixels)
    c1 = requantize(_conv(x, params["conv1_w"], params["conv1_b"]),
                    params["conv1_multiplier"], relu=True)
    n, channels, _, _ = c1.shape
    pooled = c1.reshape(n, channels, 31, 2, 31, 2).max(axis=(3, 5))
    c2 = requantize(_conv(pooled, params["conv2_w"], params["conv2_b"]),
                    params["conv2_multiplier"], relu=True)
    sums = c2.astype(np.int32).sum(axis=(2, 3))
    return ((sums + 420) // 841).astype(np.int8)


def head_quantized(params, features):
    features = np.asarray(features)
    if features.dtype != np.int8 or features.ndim != 2 or features.shape[1] != 16:
        raise ValueError("Expected INT8 features with shape (N,16).")
    if np.any(features < 0):
        raise ValueError("Encoded features must be in [0,127].")
    acc = features.astype(np.int32) @ params["dense_w"].astype(np.int32).T
    acc += params["dense_b"]
    return requantize(acc, params["dense_multiplier"])


def predict_quantized(params, pixels):
    return head_quantized(params, encode_quantized(params, pixels))


def calibrate(network, images, margin=1.05, batch_size=16):
    if margin < 1:
        raise ValueError("Calibration margin must be at least one.")
    maxima = {"conv1": 0.0, "conv2": 0.0, "logits": 0.0}
    for offset in range(0, len(images), batch_size):
        activations = network.forward_activations(images[offset:offset + batch_size])
        for name in maxima:
            maxima[name] = max(maxima[name], float(np.abs(activations[name]).max()))
    scales = {name: max(value * margin / 127.0, 1e-8) for name, value in maxima.items()}
    scales["input"] = INPUT_SCALE
    params = {}
    details = {}
    for layer, input_name, output_name in (
        ("conv1", "input", "conv1"),
        ("conv2", "conv1", "conv2"),
        ("dense", "conv2", "logits"),
    ):
        weight = network.params[layer + "_w"]
        weight_scale = max(float(np.abs(weight).max()) / 127.0, 1e-8)
        q_weight = np.clip(round_away(weight / weight_scale), -127, 127).astype(np.int8)
        bias_scale = scales[input_name] * weight_scale
        raw_bias = round_away(network.params[layer + "_b"].astype(np.float64) / bias_scale)
        if np.any(np.abs(raw_bias) > np.iinfo(np.int32).max):
            raise ValueError(layer + " bias exceeds INT32.")
        q_bias = raw_bias.astype(np.int32)
        ratio = bias_scale / scales[output_name]
        multiplier = int(round_away(ratio * (1 << Q_SHIFT)))
        if not 1 <= multiplier <= np.iinfo(np.int32).max:
            raise ValueError(layer + " Q24 multiplier is outside the supported range.")
        bound = 127 * np.abs(q_weight.astype(np.int64)).reshape(len(q_bias), -1).sum(axis=1)
        bound += np.abs(q_bias.astype(np.int64))
        if np.any(bound > np.iinfo(np.int32).max):
            raise ValueError(layer + " accumulator can exceed INT32.")
        params[layer + "_w"] = q_weight
        params[layer + "_b"] = q_bias
        params[layer + "_multiplier"] = np.int32(multiplier)
        details[layer] = {
            "weight_scale": weight_scale, "input_scale": scales[input_name],
            "output_scale": scales[output_name], "q24_multiplier": multiplier,
            "max_abs_accumulator_bound": int(bound.max()),
        }
    return params, {"activation_scales": scales, "layers": details,
                    "calibration_margin": margin, "calibration_examples": len(images)}


def save_quantized(path, params):
    np.savez(path, **params)


def load_quantized(path):
    with np.load(path, allow_pickle=False) as archive:
        params = {name: archive[name].copy() for name in archive.files}
    return params


def _array(name, value):
    value = np.asarray(value)
    ctype = "std::int8_t" if value.dtype == np.int8 else "std::int32_t"
    flat = value.reshape(-1)
    rows = [", ".join(str(int(v)) for v in flat[i:i + 16]) for i in range(0, len(flat), 16)]
    return ("inline constexpr " + ctype + " " + name + "[" + str(len(flat)) + "] = {\n    "
            + ",\n    ".join(rows) + "\n};\n")


def export_headers(directory, params, class_names):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    common = ('// Generated from locally trained weights. No pretrained parameters.\n'
              '#pragma once\n#include <cstdint>\n#include "model_runtime.h"\n'
              'namespace model_data {\n')
    labels = "inline constexpr const char* kClassNames[4] = {" + ", ".join(
        json.dumps(str(name)) for name in class_names) + "};\n"
    body = common + labels
    for name in ("conv1_w", "conv1_b", "conv2_w", "conv2_b", "dense_w", "dense_b"):
        body += _array(name, params[name])
    body += ("inline constexpr local_model::Model kModel = {\n"
             "    conv1_w, conv1_b, conv2_w, conv2_b, dense_w, dense_b,\n    "
             + ", ".join(str(int(params[name + "_multiplier"])) for name in ("conv1", "conv2", "dense"))
             + "\n};\n}  // namespace model_data\n")
    (directory / "model_data.h").write_text(body, encoding="utf-8")
    head = common + labels + _array("dense_w", params["dense_w"]) + _array("dense_b", params["dense_b"])
    head += ("inline constexpr local_model::HeadModel kHeadModel = {\n"
             "    dense_w, dense_b, " + str(int(params["dense_multiplier"]))
             + "\n};\n}  // namespace model_data\n")
    (directory / "model_head.h").write_text(head, encoding="utf-8")


def _metrics(logits, labels):
    predictions = np.argmax(logits, axis=1)
    matrix = np.zeros((4, 4), dtype=np.int64)
    np.add.at(matrix, (labels, predictions), 1)
    return {"accuracy": float(np.mean(predictions == labels)),
            "confusion_matrix": matrix.tolist(), "examples": len(labels)}


def export(model_path, data_path, outdir, calibration_count=256, header_dir=None):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    network = Network.load(model_path)
    with np.load(data_path, allow_pickle=False) as archive:
        x_train = archive["x_train"]
        x_test = archive["x_test"]
        y_test = archive["y_test"].astype(np.int64)
        class_names = archive["class_names"].tolist() if "class_names" in archive else [
            "READY", "BUSY", "DONE", "OTHER"]
        raw_metadata = archive["metadata_json"].item() if "metadata_json" in archive else "{}"
    metadata = json.loads(str(raw_metadata))
    # Calibration uses training examples only. The final test split stays separate.
    count = min(calibration_count, len(x_train))
    if count < 1:
        raise ValueError("Calibration needs at least one training example.")
    indices = np.random.default_rng(17).choice(len(x_train), count, replace=False)
    params, calibration = calibrate(network, x_train[indices])
    save_quantized(outdir / "model_int8.npz", params)
    export_headers(header_dir or outdir.parent / "model_data", params, class_names)
    save_quantized(outdir / "model_encoder.npz", {k: v for k, v in params.items() if not k.startswith("dense")})
    save_quantized(outdir / "model_head.npz", {k: v for k, v in params.items() if k.startswith("dense")})
    pixels = np.clip(round_away(x_test * 255.0), 0, 255).astype(np.uint8)
    all_quantized = []
    all_float = []
    for offset in range(0, len(pixels), 16):
        all_quantized.append(predict_quantized(params, pixels[offset:offset + 16]))
        all_float.append(network.forward(x_test[offset:offset + 16]))
    logits = np.concatenate(all_quantized)
    float_logits = np.concatenate(all_float)
    golden_dir = outdir / "golden"
    golden_dir.mkdir(exist_ok=True)
    selected = []
    for label in range(4):
        candidates = np.flatnonzero(y_test == label)
        selected.extend(candidates[:4].tolist())
    goldens = []
    for index in selected:
        image_file = f"image_{index:03d}.raw"
        feature_file = f"features_{index:03d}.raw"
        pixels[index].tofile(golden_dir / image_file)
        features = encode_quantized(params, pixels[index:index + 1])
        features[0].tofile(golden_dir / feature_file)
        goldens.append({"image_file": image_file, "feature_file": feature_file,
                        "true_label": int(y_test[index]),
                        "expected_class": int(np.argmax(logits[index])),
                        "expected_logits": logits[index].astype(int).tolist()})
    (golden_dir / "expected.json").write_text(json.dumps(goldens, indent=2) + "\n", encoding="utf-8")
    report = {
        "hardware_target": {'board': 'Arduino UNO Q', 'variant': '4 GB RAM / 32 GB eMMC', 'default_runtime': 'UNO Q Linux: full INT8 classifier', 'optional_split': 'Linux encoder + STM32U585 MCU head; Bridge integration required', 'deployment_verified': False, 'memory_scope': 'Linux RAM and MCU SRAM are independent; raw model/arena bytes are not total runtime usage'},
        "format": "Custom per-tensor INT8 / INT32-bias / Q24-requantization, not a LiteRT file",
        "class_names": class_names, "input_shape": [1, 64, 64],
        "dataset_synthetic": metadata.get("synthetic", "unknown"),
        "training": "Locally trained from random initialization, no pretrained weights",
        "calibration": calibration, "float_test": _metrics(float_logits, y_test),
        "int8_test": _metrics(logits, y_test),
        "float_int8_class_agreement": float(np.mean(float_logits.argmax(1) == logits.argmax(1))),
        "raw_parameter_bytes": 1400, "requantization_constant_bytes": 12,
        "full_parameter_and_constant_payload_bytes": 1412,
        "head_parameter_bytes": 80, "head_requantization_constant_bytes": 4,
        "head_parameter_and_constant_payload_bytes": 84,
        "activation_arena_bytes": 38448,
        "activation_arena_scope": "Only the two fixed activation arrays; excludes input/output copies, call stack, model descriptor, SDK, Bluetooth, UART, and other firmware memory.",
        "head_mode": "Optional split: UNO Q Linux runs the encoder and sends 16 quantized features to the UNO Q STM32U585 MCU head. Requires matching encoder/scales and a separately implemented Bridge transport.",
        "limitations": [
            "Synthetic demonstration accuracy is not real desktop accuracy.",
            "OTHER contains a synthetic X symbol; it is not a validated unknown-screen detector.",
            "Arduino UNO Q Linux/STM32 memory, inference latency, Bridge transport, and mouse/keyboard control are not measured.",
            "A score or margin does not guarantee action correctness.",
        ],
    }
    for name in ("model_int8.npz", "model_encoder.npz", "model_head.npz"):
        file = outdir / name
        report[name] = {"bytes": file.stat().st_size, "sha256": hashlib.sha256(file.read_bytes()).hexdigest()}
    (outdir / "quantization_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Exported full and head-only INT8 models; synthetic test accuracy "
          f"{report['int8_test']['accuracy']:.3f}; class agreement "
          f"{report['float_int8_class_agreement']:.3f}.", flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="artifacts/model.npz")
    parser.add_argument("--data", default="data/demo_dataset.npz")
    parser.add_argument("--outdir", default="artifacts")
    parser.add_argument("--calibration-count", type=int, default=256)
    parser.add_argument("--header-dir", type=Path)
    args = parser.parse_args(argv)
    if args.calibration_count < 1:
        parser.error("--calibration-count must be positive.")
    export(args.model, args.data, args.outdir, args.calibration_count, args.header_dir)


if __name__ == "__main__":
    main()
