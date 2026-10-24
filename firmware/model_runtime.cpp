#include "model_runtime.h"
#include "fixed_math.h"

#include <limits>

namespace local_model {
namespace {

bool biases_fit(const std::int32_t* biases, std::size_t count,
                std::int64_t max_dot_product) {
    const std::int64_t maximum = std::numeric_limits<std::int32_t>::max();
    for (std::size_t i = 0; i < count; ++i) {
        const std::int64_t bias = biases[i];
        const std::int64_t magnitude = bias < 0 ? -bias : bias;
        if (magnitude + max_dot_product > maximum) {
            return false;
        }
    }
    return true;
}

void conv3_valid(const std::int8_t* input,
                 std::int8_t* output,
                 int input_channels, int input_side, int output_channels,
                 const std::int8_t* weights,
                 const std::int32_t* biases,
                 std::int32_t multiplier) {
    const int output_side = input_side - 2;
    const int input_plane = input_side * input_side;
    const int output_plane = output_side * output_side;
    for (int oc = 0; oc < output_channels; ++oc) {
        for (int oy = 0; oy < output_side; ++oy) {
            for (int ox = 0; ox < output_side; ++ox) {
                std::int32_t accumulator = biases[oc];
                for (int ic = 0; ic < input_channels; ++ic) {
                    const int input_channel_offset = ic * input_plane;
                    const int weight_offset = (oc * input_channels + ic) * 9;
                    for (int ky = 0; ky < 3; ++ky) {
                        const int input_row = input_channel_offset
                                            + (oy + ky) * input_side + ox;
                        for (int kx = 0; kx < 3; ++kx) {
                            accumulator += static_cast<std::int32_t>(
                                input[input_row + kx])
                                * static_cast<std::int32_t>(
                                    weights[weight_offset + ky * 3 + kx]);
                        }
                    }
                }
                output[oc * output_plane + oy * output_side + ox] =
                    requantize(accumulator, multiplier, true);
            }
        }
    }
}

void max_pool_2(const std::int8_t* input, std::int8_t* output) {
    constexpr int channels = 8;
    constexpr int input_side = 62;
    constexpr int output_side = 31;
    for (int c = 0; c < channels; ++c) {
        for (int oy = 0; oy < output_side; ++oy) {
            for (int ox = 0; ox < output_side; ++ox) {
                const int index = c * input_side * input_side
                                  + (oy * 2) * input_side + ox * 2;
                std::int8_t maximum = input[index];
                for (int ky = 0; ky < 2; ++ky) {
                    for (int kx = 0; kx < 2; ++kx) {
                        const std::int8_t value =
                            input[index + ky * input_side + kx];
                        if (value > maximum) {
                            maximum = value;
                        }
                    }
                }
                output[c * output_side * output_side
                       + oy * output_side + ox] = maximum;
            }
        }
    }
}

void global_average(const std::int8_t* input, std::int8_t* output) {
    constexpr int plane = 29 * 29;
    for (int c = 0; c < 16; ++c) {
        std::int32_t sum = 0;
        for (int i = 0; i < plane; ++i) {
            sum += input[c * plane + i];
        }
        // Input follows ReLU, so it is nonnegative.
        output[c] = static_cast<std::int8_t>((sum + plane / 2) / plane);
    }
}

void dense(const Model& model, const std::int8_t* input, std::int8_t* output) {
    for (int oc = 0; oc < 4; ++oc) {
        std::int32_t accumulator = model.dense_b[oc];
        for (int ic = 0; ic < 16; ++ic) {
            accumulator += static_cast<std::int32_t>(input[ic])
                           * static_cast<std::int32_t>(
                               model.dense_w[oc * 16 + ic]);
        }
        output[oc] = requantize(accumulator, model.dense_multiplier, false);
    }
}

}  // namespace

bool model_is_valid(const Model& model) {
    if (!model.conv1_w || !model.conv1_b || !model.conv2_w || !model.conv2_b
        || !model.dense_w || !model.dense_b
        || model.conv1_multiplier <= 0
        || model.conv2_multiplier <= 0
        || model.dense_multiplier <= 0) {
        return false;
    }
    // Activations are [0,127]. Using abs(weight)<=128 handles every int8 value.
    constexpr std::int64_t max_product = 127 * 128;
    return biases_fit(model.conv1_b, 8, 9 * max_product)
           && biases_fit(model.conv2_b, 16, 72 * max_product)
           && biases_fit(model.dense_b, 4, 16 * max_product);
}

std::int8_t requantize(std::int32_t accumulator,
                      std::int32_t multiplier,
                      bool relu) {
    return detail::requantize_value(accumulator, multiplier, relu);
}
int infer(const Model& model, Arena& arena,
          const std::uint8_t* input_pixels,
          std::int8_t* output_logits) {
    if (!input_pixels || !output_logits || !model_is_valid(model)) {
        return -1;
    }

    for (std::size_t i = 0; i < kInputBytes; ++i) {
        arena.a[i] = static_cast<std::int8_t>(
            (static_cast<std::uint32_t>(input_pixels[i]) * 127U + 127U) / 255U);
    }
    conv3_valid(arena.a, arena.b, 1, 64, 8,
                model.conv1_w, model.conv1_b, model.conv1_multiplier);
    max_pool_2(arena.b, arena.a);
    conv3_valid(arena.a, arena.b, 8, 31, 16,
                model.conv2_w, model.conv2_b, model.conv2_multiplier);
    global_average(arena.b, arena.a);
    dense(model, arena.a, arena.b);

    int best = 0;
    for (std::size_t i = 0; i < kClassCount; ++i) {
        output_logits[i] = arena.b[i];
        if (arena.b[i] > arena.b[best]) {
            best = static_cast<int>(i);
        }
    }
    return best;
}

}  // namespace local_model
