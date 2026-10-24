#include "model_runtime.h"
#include "fixed_math.h"

#include <limits>

namespace local_model {

bool head_model_is_valid(const HeadModel& model) {
    if (!model.dense_w || !model.dense_b || model.dense_multiplier <= 0) {
        return false;
    }
    constexpr std::int64_t max_dot_product = 16 * 128 * 128;
    const std::int64_t limit = std::numeric_limits<std::int32_t>::max();
    for (std::size_t i = 0; i < kClassCount; ++i) {
        const std::int64_t bias = model.dense_b[i];
        const std::int64_t magnitude = bias < 0 ? -bias : bias;
        if (magnitude + max_dot_product > limit) {
            return false;
        }
    }
    return true;
}

int infer_head(const HeadModel& model, const std::int8_t* features,
               std::int8_t* output_logits) {
    if (!features || !output_logits || !head_model_is_valid(model)) {
        return -1;
    }
    int best = 0;
    for (int oc = 0; oc < 4; ++oc) {
        std::int32_t accumulator = model.dense_b[oc];
        for (int ic = 0; ic < 16; ++ic) {
            accumulator += static_cast<std::int32_t>(features[ic])
                           * static_cast<std::int32_t>(
                               model.dense_w[oc * 16 + ic]);
        }
        output_logits[oc] =
            detail::requantize_value(accumulator, model.dense_multiplier, false);
        if (output_logits[oc] > output_logits[best]) {
            best = oc;
        }
    }
    return best;
}

}  // namespace local_model
