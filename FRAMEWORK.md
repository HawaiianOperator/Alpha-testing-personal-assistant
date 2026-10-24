> **Hardware update:** The current target is Arduino UNO Q 4 GB / 32 GB. This earlier WROVER design is historical. See [the current implementation](personal-assistant-model/README.md) for supported components and limits.

# Personal Assistant: ESP32-WROVER Framework

A privacy-focused local desktop assistant built from a computer companion and an **ESP32-WROVER v3.0** board.

**Status: rough design, not implemented firmware.** The stated **1 GB limit applies to storage/model size**, not available RAM. The user-provided Freenove spec sheet lists 520 KB SRAM, 4 MB flash, and 8 MB PSRAM. Firmware must still verify the detected capacities and available allocation regions. No live model, firmware, or desktop action loop has been validated.

## Confirmed requirements

- Open-source AI and local processing. Design and train the models from random initialization, without pretrained weights.
- Computer software can capture the selected screen area.
- Software communicates with the microcontroller.
- MCU mode performs its own small-model inference and chooses bounded actions.
- An independent desktop mode runs a different local model on the computer.
- Mouse and keyboard control, multiple workflow profiles, and broad host OS portability.
- Explicit, manually calculated memory allocation budgets, checked against measurements.
- GitHub source management and Docker development/build tooling.
- Storage/model-size cap: 1 GB. Use a conservative decimal 1,000,000,000-byte cap until exact judging rules specify otherwise. Whether build tools/images count toward this cap remains to be confirmed.

## Hardware reality

The user-provided spec sheet identifies a Freenove ESP32 board with a dual-core 32-bit processor running at 80-240 MHz, 520 KB internal SRAM, 4 MB flash, 8 MB physical PSRAM, Bluetooth 4.2 (LE), onboard camera, and a memory card slot. These are the design's stated hardware capacities, to be checked by the initial firmware probe.

| Region | Stated capacity | Role in this project |
| --- | --- | --- |
| Internal SRAM | 520 KB | SDK, task stacks, communication, latency-sensitive state; not all is available to the application |
| Flash | 4 MB | Bootloader, partitions, firmware, and any embedded model/assets |
| Physical PSRAM | 8 MB | Larger model working/input buffers where supported; mapping constraints apply |
| Optional memory card | Card-dependent | Additional local storage, subject to the 1 GB project cap; not extra working RAM |

The onboard camera is not required for host screen capture and should remain uninitialized in the first prototype. This avoids camera buffers and peripheral contention during memory measurements.

Classic ESP32 maps up to 4 MiB of external PSRAM directly. A board with more physical PSRAM may require bank switching for the additional memory, so physical PSRAM and a contiguous tensor arena are separate limits.

The original ESP32 does not provide the native USB keyboard/mouse peripheral available on ESP32-S2/S3. A development board's USB-to-UART bridge can carry observations and programming data, but does not turn the ESP32 into a USB keyboard or mouse.

Possible input execution paths:

1. **WROVER Bluetooth HID:** send screen observations over USB serial; the WROVER sends keyboard/mouse reports over paired Bluetooth. This keeps action selection and input generation on the board, but requires compatible host Bluetooth.
2. **External USB HID bridge:** the WROVER directs a second USB-capable controller. Use this if wired HID is required; it introduces extra hardware and its own firmware.
3. **Companion execution:** the WROVER sends action packets over serial and the computer companion performs native input. This uses the current hardware with a wired connection, but input execution resides in the host software.

Choose and test one input path before claiming cross-platform support. The initial WROVER design proposal is USB serial observations plus Bluetooth HID output; this choice is not yet a validated implementation.

## Device-mode data flow

```text
COMPUTER                                     ESP32-WROVER
Task input + selected-window capture
             |
Crop/resize to a fixed observation format
             |
             +---- USB-to-UART data -------> Bounded receive buffer
                                             Small local AI model
                                             Decision/control logic
                                             Bounded action queue
Computer receives mouse/keyboard <---------- Bluetooth HID (proposed)
             |
Capture fresh screen state ---- feedback loop ---->
```

Capture and deterministic preprocessing need no host AI in MCU mode. The board receives observations rather than directly reading the display.

## Independent modes

| Mode | Observation | Decision | Execution |
| --- | --- | --- | --- |
| MCU | Host captures a bounded region or structured UI data | Small model on WROVER | Selected WROVER input path |
| Desktop | Host captures locally | Different local PC model | Native executor, or optional WROVER input controller |

Each target has its own inference path and memory budget. The models need not have equal capabilities. Only one mode owns the action executor at a time.

## Proposed repository layout

```text
Personal-assistant/
  host/
    capture/          # Windows, macOS, Linux capture backends
    observations/     # Crop, resize, fixed format, coordinate mapping
    local_ai/         # Independent desktop model
    transport/        # Framed USB-to-UART communication
    executor/         # Native input for desktop or companion execution
    app/              # Task entry, mode selector, visible status
  firmware/
    board/            # WROVER initialization, memory detection/placement
    inference/        # Small model, preprocessing, fixed tensor arena
    transport/        # UART receive/transmit, frame parser
    hid/              # Bluetooth HID; optional external bridge backend
    actions/          # Bounded action queue
    control/          # Sessions, feedback, timeouts, stop handling
  shared/
    protocol/         # Message formats, limits, versions
    workflows/        # Supported tasks and OS-specific profiles
  models/
    manifests/        # Versions, licenses, hashes, shapes, quantization
  memory/
    calculations/     # Manual byte counts, lifetimes, alignment
    reports/          # Linker maps and measured peaks
  tests/
    protocol/
    memory/
    workflows/
  docker/             # Repeatable development and build tools
```

The tree is a proposal; these modules have not yet been created or tested.

## Suggested stack

- Host companion: Python initially, with native capture and input adapters for each OS.
- Firmware: C/C++ using ESP-IDF, target esp32.
- Observation transport: the board's USB-to-UART bridge with a bounded framed protocol.
- Input: ESP-IDF Bluetooth HID APIs for the proposed WROVER path. TinyUSB is relevant only to a separate compatible USB bridge/target.
- MCU inference: LiteRT for Microcontrollers through a compatible ESP32 port, with fixed input dimensions and a fixed tensor arena.
- Desktop inference: a separately selected open-source local model/runtime obeying the applicable storage cap. Disable runtime downloads and cloud calls.
- Docker: compiler/toolchain, protocol tests, and memory-report tooling. Native host capture and device integration run on the computer. Docker does not run on the ESP32.

## Communication protocol

Use a length-delimited binary frame with explicitly sized fields:

```text
version | message_type | session_id | sequence | payload_length | payload | checksum
```

Define OBSERVATION, ACTION, ACK, STOP, ERROR, and CAPABILITIES messages. CAPABILITIES reports actual detected memory and supported input paths. A checksum detects corruption; it is not authentication or encryption.

Observation metadata includes its ID, dimensions, pixel format, crop origin, and desktop coordinate mapping. Actions identify the observation used for their decision.

- Declare maximum payload size, buffer count, queue depth, and outstanding action count.
- Receive image data in bounded chunks; avoid a redundant whole-image transport copy.
- Reject oversized messages, duplicate actions, expired observations, and stale sessions.
- Perform one bounded action and request a new observation to confirm its result.
- Define disconnect and timeout behavior, including release of held keys/buttons.
- Keep both models from issuing conflicting input.

## Manual memory budget

The 1 GB storage/model cap is an upper bound on artifacts. A model below this cap can still be far too large for the WROVER's flash, PSRAM, compute capacity, or tensor arena.

Track internal RAM, directly mapped PSRAM, banked PSRAM if present, flash, and any external storage separately. PC RAM and GPU memory are separate budgets again.

| Allocation | Calculation / evidence |
| --- | --- |
| One 96 by 96 grayscale observation | 96 * 96 * 1 = 9,216 bytes |
| Two such observations, if needed | 2 * 9,216 = 18,432 bytes |
| UART RX/TX and framing | Declared capacities plus parser state |
| Action queue | Queue capacity * actual sizeof(Action), including padding |
| Model weights | Actual file bytes; record flash placement and any RAM copies |
| Tensor arena | Peak live tensors, scratch, operator state, alignment |
| Bluetooth/SDK buffers | SDK configuration, linker report, runtime measurements |
| Task and interrupt stacks | Allocated stack bytes, verified high-water measurements |
| Application state | Actual linked structures and static arrays |
| Reserve | Explicit unallocated margin |

Use fixed shapes, buffers, and queue capacities. Prefer static application allocation or initialization-time allocation; audit SDK/runtime heap use before claiming zero dynamic allocation.

Check manual calculations against linker maps, .data/.bss, sizeof, arena reports, stack watermarks, and worst-case execution. Do not add static buffers again after they are already counted in linked sections. Leave memory for firmware and Bluetooth even if PSRAM appears plentiful.

Store model weights locally. On-board flash has its own capacity/partition limit; using an SD card, if present or added, supplies storage rather than working RAM.

For the PC model, count weights, context cache, activations, image processing, and runtime overhead. OS/runtime allocation remains local.

## Privacy requirements

- No cloud inference or external API dependency during operation.
- Capture only the user-selected region/window required by the task.
- Keep a bounded, transient observation history.
- No screen-content/prompt logging by default.
- Keep credentials, screenshots, and personal documents out of Git history.
- Show capture/control status and provide an immediate stop control.
- Document the data received by every process/device.
- For Bluetooth input, explicitly pair with the chosen host and test the intended link security.
- Wi-Fi can remain disabled in the proposed wired-observation architecture.

The capture app and host OS still see the screen. Hardware separation does not conceal it from them or establish a general security guarantee.

## First milestone

1. Build a WROVER probe that reports internal RAM, flash, physical and usable PSRAM, and largest suitable allocation regions.
2. Verify USB serial communication and the selected input path independently.
3. Use a test window with two known visual states. Send a fixed small grayscale observation to the board.
4. Run a real small classifier on the WROVER; choose and perform one action.
5. Capture a fresh observation and confirm the new state.
6. Run the same task through the independent desktop model.
7. Record manual and measured memory/storage totals.

General Muse-level screen understanding and unrestricted automation are beyond the initial MCU scope. Demonstrate one learned visual task before expanding purposes.

## References

- [ESP32 datasheet](https://www.espressif.com/sites/default/files/documentation/esp32_datasheet_en.pdf)
- [ESP-WROVER-KIT v3 guide](https://docs.espressif.com/projects/esp-dev-kits/en/latest/esp32/esp-wrover-kit/user_guide_v3.html) (applies specifically to Espressif's kit, not every WROVER-branded board)
- [ESP32 PSRAM mapping](https://docs.espressif.com/projects/esp-idf/en/stable/esp32/api-reference/system/himem.html)
- [Espressif USB-capable chip distinction](https://docs.espressif.com/projects/arduino-esp32/en/latest/api/usb.html)
- [ESP-IDF Bluetooth HID device example](https://github.com/espressif/esp-idf/tree/master/examples/bluetooth/esp_hid_device)
- [LiteRT for Microcontrollers](https://developers.google.com/edge/litert/microcontrollers/overview)
- [TFLM memory planning](https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/docs/memory_management.md)

## Custom model from scratch

The team will design its own architecture and train from random initialization. See [MODEL_PLAN.md](MODEL_PLAN.md) for a proposed four-state 64 by 64 grayscale CNN, hand-calculated parameter/activation payloads, and the local training/export workflow. Training is proposed on the local computer; MCU-mode inference runs on WROVER. This is a design plan, not a trained or measured model.
