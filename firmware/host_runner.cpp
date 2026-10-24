// Host verification tool; only model_runtime.cpp is the embedded inference core.
#include "model_runtime.h"
#include "model_data.h"

#include <chrono>
#include <cstdio>

namespace {
local_model::Arena arena;
std::uint8_t pixels[local_model::kInputBytes];
std::int8_t logits[local_model::kClassCount];
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "Usage: model_runner <4096-byte grayscale image>\n");
        return 2;
    }
    std::FILE* input = std::fopen(argv[1], "rb");
    if (!input) {
        std::fprintf(stderr, "Cannot open input image.\n");
        return 2;
    }
    const std::size_t read =
        std::fread(pixels, 1, local_model::kInputBytes, input);
    const int extra = std::fgetc(input);
    const bool read_error = std::ferror(input) != 0;
    std::fclose(input);
    if (read != local_model::kInputBytes || extra != EOF || read_error) {
        std::fprintf(stderr, "Input must contain exactly 4096 grayscale bytes.\n");
        return 2;
    }

    const auto start = std::chrono::steady_clock::now();
    const int predicted =
        local_model::infer(model_data::kModel, arena, pixels, logits);
    const auto end = std::chrono::steady_clock::now();
    if (predicted < 0) {
        std::fprintf(stderr, "Invalid model configuration.\n");
        return 3;
    }
    const auto micros =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::printf(
        "{\"class_id\":%d,\"logits\":[%d,%d,%d,%d],"
        "\"arena_bytes\":%zu,\"input_bytes\":%zu,\"elapsed_us\":%lld}\n",
        predicted, static_cast<int>(logits[0]), static_cast<int>(logits[1]),
        static_cast<int>(logits[2]), static_cast<int>(logits[3]),
        sizeof(arena), sizeof(pixels), static_cast<long long>(micros));
    return 0;
}
