> **Hardware update:** The current target is Arduino UNO Q 4 GB / 32 GB. This earlier WROVER design is historical. See [the current implementation](personal-assistant-model/README.md) for supported components and limits.

# Custom model plan: train from scratch for ESP32-WROVER

## Scope and assumptions

Build the model architecture and learn its weights from random initialization, using local training data. No pretrained weights or remote inference are part of this plan. Open-source training/inference frameworks can still supply the numerical operations.

Training is proposed on the local computer; MCU-mode preprocessing and inference run on the ESP32-WROVER. This distinguishes training from deployment. On-board training is not assumed.

This is a model design and hand-calculation worksheet. No dataset, trained weights, converted model, measured accuracy, or board timing has been produced yet.

Hardware from the user's sheet: 520 KB SRAM, 4 MB flash, 8 MB physical PSRAM. Actual free/contiguous memory and the directly mapped PSRAM window still constrain inference. The 1 GB storage/model cap does not increase these capacities.

## First learned task

Recognize four visual states of a controlled desktop test application:

- READY: the configured action can begin.
- BUSY: wait for the current operation.
- DONE: the task completed.
- OTHER: an unsupported view.

The neural network recognizes state. A separate bounded controller uses that state, the selected workflow, and the action history to decide the next action. A READY prediction must not repeatedly replay the starting click.

This is a first learned task, not general understanding of arbitrary applications, text, or instructions.

## Proposed architecture

Fixed input: one 64 by 64 grayscale crop, batch size one. Convolutions use stride one and valid padding. Max pooling uses a 2 by 2 window and stride two. ReLU can be fused into each convolution. The final layer returns four logits.

| Stage | Output shape | Int8 output bytes | Weight count | Bias count |
| --- | --- | ---: | ---: | ---: |
| Input | 64 * 64 * 1 | 4,096 | 0 | 0 |
| Conv 3 * 3, 8 filters | 62 * 62 * 8 | 30,752 | 72 | 8 |
| MaxPool 2 * 2 | 31 * 31 * 8 | 7,688 | 0 | 0 |
| Conv 3 * 3, 16 filters | 29 * 29 * 16 | 13,456 | 1,152 | 16 |
| Global average pooling | 16 | 16 | 0 | 0 |
| Dense, 4 outputs | 4 | 4 | 64 | 4 |

TensorFlow/Keras or PyTorch can be used for local training. Select and pin the export route after checking every converted operation against the chosen MCU runtime.

## Hand-calculated parameter payload

Convolution weight count:

    kernel height * kernel width * input channels * output channels

First convolution:

    weights = 3 * 3 * 1 * 8 = 72
    biases = 8

Second convolution:

    weights = 3 * 3 * 8 * 16 = 1,152
    biases = 16

Dense layer:

    weights = 16 * 4 = 64
    biases = 4

Totals:

    weights = 72 + 1,152 + 64 = 1,288
    biases = 8 + 16 + 4 = 28
    parameters = 1,288 + 28 = 1,316

Float32 parameter payload:

    1,316 * 4 = 5,264 bytes

Quantized parameter payload, assuming int8 weights and int32 biases:

    1,288 * 1 + 28 * 4 = 1,400 bytes

These are raw parameter bytes. Add model graph metadata, scales/zero points, tensor descriptions, constant axes, operator options, and alignment to obtain the actual file size. Inspect the converted file rather than labeling it a 1,400-byte model.

The int32 bias assumption follows the LiteRT int8 operator specification.

## Hand-calculated activation payload

Holding every listed tensor in a separate buffer:

    4,096 + 30,752 + 7,688 + 13,456 + 16 + 4 = 56,012 bytes

For an illustrative sequential implementation, the largest adjacent live input/output pair is the first convolution output and the pooling output:

    30,752 + 7,688 = 38,440 bytes

A hypothetical two-buffer arrangement can alternate these stages:

- Buffer A: input, pooling result, and global-pooling result; maximum 7,688 bytes.
- Buffer B: convolution results and final logits; maximum 30,752 bytes.

With each capacity rounded up to 16-byte alignment:

    A = ceil(7,688 / 16) * 16 = 7,696
    B = ceil(30,752 / 16) * 16 = 30,752
    total = 38,448 bytes

This is an activation-payload plan, NOT a proven LiteRT Micro tensor-arena size. The actual runtime's lifetimes, persistent operator data, scratch, temporary allocations, and alignment may require more. A static offline plan must match the exported graph and runtime.

If received pixels already occupy the model input buffer, do not count them a second time. Retaining another full input crop adds 4,096 bytes. Chunked transport can avoid a second full-size receive buffer.

The convolution/dense computation contains approximately 1,245,664 multiply-accumulates per inference, excluding pooling and preprocessing. This count does not establish latency; measure on the selected board.

## Dataset and local training

1. Capture only the controlled application region and label its state.
2. Include varied positions, scaling, themes, lighting/rendering changes, and unsupported views relevant to the demonstration.
3. Split by capture session/layout, keeping near-identical adjacent frames together. Reserve a final held-out test set.
4. Train from random weights and track per-class errors. Do not set a guaranteed accuracy or training-data quantity in advance.
5. Export a fully integer model with representative calibration examples drawn from the training/calibration partition.
6. Verify converted operators and input/output types; reject a conversion that leaves unsupported floating-point operations.
7. Compare float and quantized model results on held-out data, including unsupported screens.
8. Measure end-to-end false actions, missed states, latency, and peak memory on the board.

Training data, captures, and labels remain local. GitHub should contain code, manifests, and non-sensitive synthetic fixtures. Publish captured images or trained weights only after deciding which data/assets are appropriate to share.

## Firmware/controller integration

Host capture -> fixed preprocessing -> framed USB serial -> MCU inference -> state/controller decision -> selected input executor -> fresh observation.

For this WROVER, the proposed hardware input path is Bluetooth HID. USB keyboard/mouse output requires a separate compatible bridge; the board's USB uploader is serial.

The MCU controller owns:

- The selected workflow and current expected state.
- A fixed-capacity action queue and explicit timeouts.
- Suppression of duplicate action IDs and stale observations.
- A stop/no-action path for OTHER or insufficiently validated predictions.
- Key/button release on cancellation, disconnect, or timeout.

Use validation data to choose any score/margin thresholds; a large neural-network score is not itself a correctness guarantee.

## Complete firmware memory ledger

After selecting/exporting the model, add:

- Model bytes and any RAM copies.
- Actual tensor arena, including scratch and persistent operator data.
- Fixed protocol buffers and framing state.
- Fixed action queue and control state.
- UART and Bluetooth SDK allocations.
- Task/interrupt stacks and their measured peaks.
- Alignment, linked .data/.bss, and an explicit reserve.

Budget internal SRAM and directly mapped PSRAM separately. Verify the largest suitable contiguous region, not merely total free memory. Audit dependencies before claiming zero heap allocation.

## Independent desktop model

The PC model can be trained separately from random weights using the same state labels/protocol. Its architecture can be larger, but needs its own evaluation and memory/storage budget. Do not assume the MCU and PC models have identical capability.

## Framework additions

    training/
      dataset/          # Local capture/labels; captured data ignored by Git
      model/            # Architecture and random initialization
      train/            # Local training
      quantize/         # Representative calibration and export
      evaluate/         # Session-based holdout evaluation
    memory/
      calculations/     # This hand-derived payload plan
      reports/          # Exported graph, arena, stack, and board measurements

Docker can package local training/export and firmware build tools. It runs on the development computer. Build-tool/image inclusion in the 1 GB limit still needs to follow the hackathon's precise rules.

## References

- [Keras: image classification from scratch](https://keras.io/examples/vision/image_classification_from_scratch/)
- [LiteRT int8 operator specification](https://developers.google.com/edge/litert/quantization/quantization_spec)
- [Full integer quantization and calibration](https://developers.google.com/edge/litert/conversion/tensorflow/quantization/post_training_quantization)
- [TFLM arena and offline allocation planning](https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/docs/memory_management.md)
