#ifndef LOCAL_MODEL_FIXED_MATH_H
#define LOCAL_MODEL_FIXED_MATH_H

#include "model_runtime.h"

namespace local_model {
namespace detail {

inline std::int8_t requantize_value(std::int32_t accumulator,
                                    std::int32_t multiplier,
                                    bool relu) {
    const std::int64_t product =
        static_cast<std::int64_t>(accumulator) * multiplier;
    const std::int64_t magnitude = product < 0 ? -product : product;
    constexpr std::int64_t half = std::int64_t{1} << (kMultiplierShift - 1);
    const std::int64_t rounded = (magnitude + half) >> kMultiplierShift;
    std::int64_t value = product < 0 ? -rounded : rounded;
    const std::int64_t lower = relu ? 0 : -127;
    if (value < lower) {
        value = lower;
    }
    if (value > 127) {
        value = 127;
    }
    return static_cast<std::int8_t>(value);
}

}  // namespace detail
}  // namespace local_model
#endif
