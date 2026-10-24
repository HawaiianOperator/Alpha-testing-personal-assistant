> **Current target: UNO Q.** See [UNO_Q.md](../UNO_Q.md). These are portable native-tested components, not a verified UNO Q/STM32 deployment. The proposed Linux/STM32 split requires board integration and measurement.

# Fixed-memory embedded inference core

This is a portable C++17 implementation of the custom 64 by 64 grayscale CNN. It performs integer inference with exported quantized parameters. It does not train a model, capture a screen, initialize UART/Bluetooth, or send keyboard/mouse actions.

## Interface

```cpp
#include "model_runtime.h"
#include "model_data.h"

static local_model::Arena arena;
static uint8_t pixels[4096];
static int8_t logits[4];

int state = local_model::infer(model_data::kModel, arena, pixels, logits);
```

Only one inference may use an arena at a time. The input/output pointers must refer to separate buffers that do not overlap the arena. Model arrays must have the exact declared lengths and stay valid throughout inference. The function returns -1 for invalid pointers/configuration and otherwise returns the first maximum logit index.

The model is declared as six constant pointers and three Q24 multipliers in `local_model::Model`. Generated `model_data.h` must define `model_data::kModel`.

## Numerical contract

- Pixels: exactly 4,096 uint8 values, row-major grayscale.
- Convert to int8 [0,127]: `(pixel * 127 + 127) / 255`.
- Tensor order: NCHW, batch one. Convolution weights: OIHW; dense weights: OI.
- Conv 3 by 3 valid, 8 filters, ReLU.
- Max pool 2 by 2, stride two.
- Conv 3 by 3 valid, 16 filters, ReLU.
- Global average over each 29 by 29 feature plane; round positive mean to nearest integer.
- Dense 16 to 4, output int8 logits clamped to [-127,127].
- Biases and accumulators: int32. Requantization multiply: int64.
- Each affine stage has one positive per-tensor Q24 multiplier. Requantization rounds nearest with exact half values away from zero, then clamps.
- No softmax is used. Logits are integer score outputs; class labels come from the model manifest.

The exporter must ensure int32 accumulator bounds and calibrate stage scales. Runtime validation conservatively checks all biases plus worst-case dot products before inference. The core does not verify the quantization/calibration quality.

## Hand-calculated resident buffers

| Allocation | Bytes |
| --- | ---: |
| Arena A, 16-byte aligned | 7,696 |
| Arena B, 16-byte aligned | 30,752 |
| Total activation arena | **38,448** |
| Caller input, if retained separately | 4,096 |
| Caller output logits | 4 |

The static assertion checks the actual C++ arena layout. These figures are not a whole-firmware RAM budget. Add model metadata/pointers, call/interrupt stacks, UART/Bluetooth SDK allocations, task state, and explicit reserve. Weights and biases may remain in mapped flash if the platform/linker supports it; verify the target linker map. Code storage and firmware partitions also consume flash.

The core uses no heap allocation, recursion, variable-length arrays, exceptions, or floating-point arithmetic. The host CLI uses the C runtime for file access; that CLI is not the embedded application.

The runtime is deliberately scalar and prioritizes inspectable memory and correctness. Its latency has to be measured on the UNO Q; host timing is not a substitute.

## Build a self-test

With GCC/MinGW available:

```powershell
g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic firmware/model_runtime.cpp firmware/self_test.cpp -o runtime_self_test
./runtime_self_test
```

Or configure this directory using CMake and run CTest.

## Check exported weights on a computer

From the project root after quantized export:

```powershell
g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic -Imodel_data firmware/model_runtime.cpp firmware/host_runner.cpp -o model_runner
./model_runner model_data/example.raw
```

The runner accepts exactly 4,096 raw grayscale bytes. It prints a JSON class index, integer logits, arena/input size, and host elapsed microseconds. Compare its logits to the Python integer reference on multiple held-out images before deploying.

## Arduino UNO Q integration

`integration_example.cpp` is an entry-point example, not a complete Arduino UNO Q App Lab/STM32 project. Connect a complete observation receiver to `model_process_observation`, then pass its prediction to the workflow controller. A prediction alone must not directly repeat clicks without action/session checks.

Keep a single static arena in a measured memory region. UNO Q Linux RAM and STM32 SRAM are separate budgets. Start with the full native classifier on Linux; an optional MCU head needs matching encoder features and a separately implemented Arduino Bridge transport.

No hardware has been flashed or timed by this implementation.

## Head-only MCU deployment

For the smallest MCU payload, compile **only** `head_runtime.cpp` and the generated `model_head.h` parameter header. Use `head_integration_example.cpp` as the board entry-point pattern. Do not compile the full runtime or include full `model_data.h` for this target.

```cpp
#include "model_runtime.h"
#include "model_head.h"

static int8_t features[16];
static int8_t logits[4];
int state = local_model::infer_head(model_data::kHeadModel, features, logits);
```

This is a partial-model deployment: **UNO Q Linux (or an optional computer companion) must run the matching calibrated image encoder** and supply its sixteen quantized GAP features. Raw pixels cannot be passed directly to this head. All processing can remain local, but this mode does not put the entire vision model on the MCU. Feature order, quantization scales, model version, and class ordering must match the exported head.

The head uses:

| Item | Raw bytes |
| --- | ---: |
| Dense weights, 4 * 16 int8 | 64 |
| Dense biases, 4 int32 | 16 |
| Q24 multiplier, one int32 | 4 |
| Mathematical parameter payload | **84** |
| Caller feature vector | 16 |
| Caller output vector | 4 |
| CNN activation arena | **0** |

This excludes pointer metadata, alignment, call stacks, communication state, and the Linux or Zephyr/Arduino runtime. In particular, `HeadModel` contains two pointers plus the multiplier; its multiplier is already in the 84-byte payload, so do not add it twice. On a typical 32-bit ABI, parameters can occupy 80 bytes of arrays plus a 12-byte descriptor before extra alignment, rather than claiming 84 bytes is the complete firmware footprint. Verify the actual UNO Q STM32 linker map and stack high-water marks.

The head accepts signed int8 features and accumulates sixteen products per class. It uses the same Q24 rounding and output saturation as the full network. No activation arena, heap allocation, or floating-point arithmetic is needed in the head inference core.

Build the independent head test:

```powershell
g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic firmware/head_runtime.cpp firmware/head_self_test.cpp -o head_self_test
./head_self_test
```

After export, build its host verification runner:

```powershell
g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic -Imodel_data firmware/head_runtime.cpp firmware/head_runner.cpp -o head_runner
./head_runner model_data/example.features.raw
```

The head runner accepts exactly sixteen signed int8 bytes and emits JSON logits and the class index. It links without the full convolution runtime.
