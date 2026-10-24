#include "model_runtime.h"

#include <cstdint>
#include <cstdio>
#include <limits>

namespace {
local_model::Arena arena;
std::uint8_t pixels[local_model::kInputBytes] = {};
std::int8_t conv1_w[72] = {};
std::int32_t conv1_b[8] = {};
std::int8_t conv2_w[1152] = {};
std::int32_t conv2_b[16] = {};
std::int8_t dense_w[64] = {};
std::int32_t dense_b[4] = {};
constexpr std::int32_t identity = 1 << 24;

int check(bool condition, const char* message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        return 1;
    }
    return 0;
}
}

int main() {
    int failures = 0;
    failures += check(sizeof(arena) == 38448, "fixed activation memory");
    failures += check(local_model::requantize(1, 1 << 23, false) == 1,
                      "positive half rounds away from zero");
    failures += check(local_model::requantize(-1, 1 << 23, false) == -1,
                      "negative half rounds away from zero");
    failures += check(local_model::requantize(-12, identity, true) == 0,
                      "fused ReLU lower bound");
    failures += check(local_model::requantize(1000, identity, false) == 127,
                      "positive saturation");
    failures += check(local_model::requantize(-1000, identity, false) == -127,
                      "negative saturation");

    local_model::Model model{
        conv1_w, conv1_b, conv2_w, conv2_b, dense_w, dense_b,
        identity, identity, identity};
    std::int8_t logits[4] = {};
    dense_b[0] = -1;
    dense_b[1] = 2;
    dense_b[2] = 2;
    dense_b[3] = -3;
    failures += check(local_model::infer(model, arena, pixels, logits) == 1,
                      "first maximum wins a tie");
    failures += check(logits[0] == -1 && logits[1] == 2 && logits[2] == 2
                      && logits[3] == -3, "zero network preserves dense biases");

    for (int i = 0; i < 8; ++i) {
        conv1_b[i] = 1;
    }
    for (int i = 0; i < 1152; ++i) {
        conv2_w[i] = 1;
    }
    for (int i = 0; i < 16; ++i) {
        dense_w[i] = 1;
        dense_w[16 + i] = -1;
    }
    dense_b[0] = dense_b[1] = 0;
    dense_b[2] = 3;
    dense_b[3] = -3;
    failures += check(local_model::infer(model, arena, pixels, logits) == 0,
                      "constant convolution chain predicts first class");
    failures += check(logits[0] == 127 && logits[1] == -127
                      && logits[2] == 3 && logits[3] == -3,
                      "convolution, pooling, mean and dense chain");

    model.conv1_multiplier = 0;
    failures += check(local_model::infer(model, arena, pixels, logits) == -1,
                      "invalid multiplier rejected");
    model.conv1_multiplier = identity;
    conv1_b[0] = std::numeric_limits<std::int32_t>::max();
    failures += check(!local_model::model_is_valid(model),
                      "potential accumulator overflow rejected");
    conv1_b[0] = 0;
    failures += check(local_model::infer(model, arena, nullptr, logits) == -1,
                      "null input rejected");
    failures += check(local_model::infer(model, arena, pixels, nullptr) == -1,
                      "null output rejected");
    if (failures) {
        return 1;
    }
    std::printf("All fixed-memory runtime checks passed. Arena: %zu bytes.\n",
                sizeof(arena));
    return 0;
}
