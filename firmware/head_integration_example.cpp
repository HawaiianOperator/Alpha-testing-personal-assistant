// Partial-model MCU entry point. The PC must supply calibrated encoder features.
// This file does not create an activation arena or include full CNN weights.
#include "model_runtime.h"
#include "model_head.h"

extern "C" bool model_process_features(const std::int8_t* features,
                                        std::size_t byte_count,
                                        std::int8_t* output_logits,
                                        std::uint8_t* output_class) {
    if (byte_count != 16 || !output_class) {
        return false;
    }
    const int predicted =
        local_model::infer_head(model_data::kHeadModel, features, output_logits);
    if (predicted < 0) {
        return false;
    }
    *output_class = static_cast<std::uint8_t>(predicted);
    return true;
}
