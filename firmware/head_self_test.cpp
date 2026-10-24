#include "model_runtime.h"

#include <cstdint>
#include <cstdio>
#include <limits>

namespace {
std::int8_t features[16] = {};
std::int8_t weights[64] = {};
std::int32_t biases[4] = {-1, 2, 2, -3};
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
    local_model::HeadModel model{weights, biases, identity};
    std::int8_t logits[4] = {};
    failures += check(local_model::infer_head(model, features, logits) == 1,
                      "head returns first maximum in a tie");
    failures += check(logits[0] == -1 && logits[1] == 2
                      && logits[2] == 2 && logits[3] == -3,
                      "head preserves biases for zero features");
    for (int i = 0; i < 16; ++i) {
        features[i] = 1;
        weights[i] = 1;
        weights[16 + i] = -1;
    }
    biases[0] = biases[1] = 0;
    biases[2] = 3;
    failures += check(local_model::infer_head(model, features, logits) == 0,
                      "head uses all sixteen features");
    failures += check(logits[0] == 16 && logits[1] == -16
                      && logits[2] == 3 && logits[3] == -3,
                      "head applies signed weights and dense bias");

    model.dense_multiplier = 1 << 23;
    biases[2] = 1;
    biases[3] = -1;
    local_model::infer_head(model, features, logits);
    failures += check(logits[2] == 1 && logits[3] == -1,
                      "head uses symmetric half-away rounding");
    model.dense_multiplier = 0;
    failures += check(local_model::infer_head(model, features, logits) == -1,
                      "invalid multiplier rejected");
    model.dense_multiplier = identity;
    biases[0] = std::numeric_limits<std::int32_t>::max();
    failures += check(!local_model::head_model_is_valid(model),
                      "head accumulator overflow rejected");
    biases[0] = 0;
    failures += check(local_model::infer_head(model, nullptr, logits) == -1,
                      "null features rejected");
    failures += check(local_model::infer_head(model, features, nullptr) == -1,
                      "null logits rejected");
    if (failures) {
        return 1;
    }
    std::printf("All head-only checks passed. No activation arena required.\n");
    return 0;
}
