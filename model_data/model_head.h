// Generated from locally trained weights. No pretrained parameters.
#pragma once
#include <cstdint>
#include "model_runtime.h"
namespace model_data {
inline constexpr const char* kClassNames[4] = {"READY", "BUSY", "DONE", "OTHER"};
inline constexpr std::int8_t dense_w[64] = {
    26, 27, -125, -28, -59, -95, -57, 33, 35, -1, 34, -15, -124, -106, -86, -16,
    57, -46, -94, 79, 51, -71, 51, 32, 19, -127, -76, -69, -59, 77, 33, -57,
    -37, 38, 21, -27, 10, 35, -7, 4, -14, -8, 16, 30, 14, 22, 16, 67,
    -69, 18, 120, -24, 41, 118, 65, -34, -48, 47, 9, 50, 113, 53, 86, 36
};
inline constexpr std::int32_t dense_b[4] = {
    23, -9, 8, -30
};
inline constexpr local_model::HeadModel kHeadModel = {
    dense_w, dense_b, 455599
};
}  // namespace model_data
