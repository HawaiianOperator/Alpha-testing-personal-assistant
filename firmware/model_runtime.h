#ifndef LOCAL_MODEL_RUNTIME_H
#define LOCAL_MODEL_RUNTIME_H

#include <cstddef>
#include <cstdint>

namespace local_model {

constexpr std::size_t kInputWidth = 64;
constexpr std::size_t kInputHeight = 64;
constexpr std::size_t kInputBytes = kInputWidth * kInputHeight;
constexpr std::size_t kClassCount = 4;
constexpr std::size_t kBufferABytes = 7696;
constexpr std::size_t kBufferBBytes = 30752;
constexpr std::size_t kArenaBytes = kBufferABytes + kBufferBBytes;
constexpr int kMultiplierShift = 24;

// All tensors are contiguous NCHW; weights use OIHW and dense uses OI.
struct Model {
    const std::int8_t* conv1_w;    // 8 * 1 * 3 * 3 = 72 values.
    const std::int32_t* conv1_b;   // 8 values.
    const std::int8_t* conv2_w;    // 16 * 8 * 3 * 3 = 1152 values.
    const std::int32_t* conv2_b;   // 16 values.
    const std::int8_t* dense_w;    // 4 * 16 = 64 values.
    const std::int32_t* dense_b;   // 4 values.
    std::int32_t conv1_multiplier;
    std::int32_t conv2_multiplier;
    std::int32_t dense_multiplier;
};

// Head-only inference is a partial-model deployment. The sixteen input features
// must come from the matching calibrated PC encoder, not directly from pixels.
struct HeadModel {
    const std::int8_t* dense_w;    // 4 * 16 = 64 values.
    const std::int32_t* dense_b;   // 4 values.
    std::int32_t dense_multiplier;
};

bool head_model_is_valid(const HeadModel& model);
int infer_head(const HeadModel& model, const std::int8_t* features,
               std::int8_t* output_logits);
struct alignas(16) Arena {
    alignas(16) std::int8_t a[kBufferABytes];
    alignas(16) std::int8_t b[kBufferBBytes];
};

static_assert(sizeof(Arena) == 38448, "Unexpected activation arena padding.");
static_assert(alignof(Arena) == 16, "The activation arena requires 16-byte alignment.");

// Checks pointers, multipliers, and conservative int32 accumulator bounds.
// Array lengths and the model's training/quantization metadata are trusted.
bool model_is_valid(const Model& model);

// Public for bit-exact reference tests. multiplier must be positive.
std::int8_t requantize(std::int32_t accumulator,
                      std::int32_t multiplier,
                      bool relu);

// Runs one image. No heap allocation, exceptions, recursion, or variable arrays.
// Converts uint8 pixels to int8 [0, 127] with (pixel * 127 + 127) / 255.
// Returns the first maximum class index; -1 means invalid configuration/input.
int infer(const Model& model, Arena& arena,
          const std::uint8_t* input_pixels,
          std::int8_t* output_logits);

}  // namespace local_model
#endif
