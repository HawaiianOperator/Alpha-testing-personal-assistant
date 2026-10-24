// Host verification tool for the partial model; no CNN activation arena.
#include "model_runtime.h"
#include "model_head.h"

#include <chrono>
#include <cstdio>

namespace {
std::int8_t features[16];
std::int8_t logits[local_model::kClassCount];
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "Usage: head_runner <16-byte quantized feature file>\n");
        return 2;
    }
    std::FILE* input = std::fopen(argv[1], "rb");
    if (!input) {
        std::fprintf(stderr, "Cannot open input feature file.\n");
        return 2;
    }
    const std::size_t read = std::fread(features, 1, sizeof(features), input);
    const int extra = std::fgetc(input);
    const bool read_error = std::ferror(input) != 0;
    std::fclose(input);
    if (read != sizeof(features) || extra != EOF || read_error) {
        std::fprintf(stderr, "Input must contain exactly 16 signed int8 features.\n");
        return 2;
    }
    const auto start = std::chrono::steady_clock::now();
    const int predicted =
        local_model::infer_head(model_data::kHeadModel, features, logits);
    const auto end = std::chrono::steady_clock::now();
    if (predicted < 0) {
        std::fprintf(stderr, "Invalid head model configuration.\n");
        return 3;
    }
    const auto micros =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::printf(
        "{\"class_id\":%d,\"logits\":[%d,%d,%d,%d],"
        "\"head_payload_bytes\":84,\"feature_bytes\":%zu,"
        "\"output_bytes\":%zu,\"elapsed_us\":%lld}\n",
        predicted, static_cast<int>(logits[0]), static_cast<int>(logits[1]),
        static_cast<int>(logits[2]), static_cast<int>(logits[3]),
        sizeof(features), sizeof(logits), static_cast<long long>(micros));
    return 0;
}
