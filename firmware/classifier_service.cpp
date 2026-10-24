// Persistent local classifier process. This is not a desktop action executor.
#include "model_runtime.h"
#include "model_data.h"
#include "task_state.h"
#include <cstdio>
#include <cstdint>
#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

namespace {
local_model::Arena arena;
local_tasks::State task;
std::uint8_t pixels[4096];
std::int8_t features[16];
std::int8_t logits[4];
std::uint8_t header[16];
char result[256];
const local_model::HeadModel head{
    model_data::kModel.dense_w, model_data::kModel.dense_b,
    model_data::kModel.dense_multiplier
};
std::uint32_t read_u32(const std::uint8_t* p) {
    return static_cast<std::uint32_t>(p[0])
        | (static_cast<std::uint32_t>(p[1]) << 8)
        | (static_cast<std::uint32_t>(p[2]) << 16)
        | (static_cast<std::uint32_t>(p[3]) << 24);
}
void clear_job() {
    const local_tasks::Region regions[] = {
        {&arena, sizeof(arena)}, {pixels, sizeof(pixels)},
        {features, sizeof(features)}, {logits, sizeof(logits)},
        {header, sizeof(header)}
    };
    local_tasks::finish_all(task, regions, sizeof(regions) / sizeof(regions[0]));
}
bool all_zero(const void* data, std::size_t size) {
    const auto* bytes = static_cast<const volatile std::uint8_t*>(data);
    for (std::size_t i = 0; i < size; ++i) if (bytes[i] != 0) return false;
    return true;
}
bool private_buffers_clear() {
    return all_zero(&arena, sizeof(arena)) && all_zero(&task, sizeof(task))
        && all_zero(pixels, sizeof(pixels)) && all_zero(features, sizeof(features))
        && all_zero(logits, sizeof(logits)) && all_zero(header, sizeof(header));
}
}

int main() {
#ifdef _WIN32
    _setmode(_fileno(stdin), _O_BINARY);
#endif
    std::setvbuf(stdin, nullptr, _IONBF, 0);
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    clear_job();
    while (true) {
        const std::size_t got = std::fread(header, 1, sizeof(header), stdin);
        if (got == 0 && std::feof(stdin)) break;
        if (got != sizeof(header) || header[0] != 'P' || header[1] != 'A'
            || header[2] != 'M' || header[3] != '1'
            || (header[4] != 1 && header[4] != 2)
            || header[5] || header[6] || header[7]) {
            clear_job();
            return 2;
        }
        const std::uint8_t mode = header[4];
        std::uint32_t task_id = read_u32(header + 8);
        const std::uint32_t payload_bytes = read_u32(header + 12);
        const std::uint32_t expected_bytes = mode == 1 ? 4096U : 16U;
        if (task_id == 0 || payload_bytes != expected_bytes
            || !local_tasks::begin(task, task_id, &mode, 1)) {
            clear_job();
            return 2;
        }
        void* input = mode == 1 ? static_cast<void*>(pixels) : static_cast<void*>(features);
        if (std::fread(input, 1, payload_bytes, stdin) != payload_bytes) {
            clear_job();
            return 2;
        }
        if (mode == 2) {
            for (const auto value : features) {
                if (value < 0) { clear_job(); return 3; }
            }
        }
        int predicted = mode == 1
            ? local_model::infer(model_data::kModel, arena, pixels, logits)
            : local_model::infer_head(head, features, logits);
        if (predicted < 0) {
            clear_job();
            return 3;
        }
        const int written = std::snprintf(result, sizeof(result),
            "{\"task_id\":%lu,\"class_id\":%d,\"logits\":[%d,%d,%d,%d],",
            static_cast<unsigned long>(task_id), predicted,
            static_cast<int>(logits[0]), static_cast<int>(logits[1]),
            static_cast<int>(logits[2]), static_cast<int>(logits[3]));
        clear_job();
        const bool cleared = private_buffers_clear();
        if (!cleared || written <= 0 || static_cast<std::size_t>(written) >= sizeof(result)) {
            local_tasks::secure_zero(result, sizeof(result));
            return 4;
        }
        std::fwrite(result, 1, static_cast<std::size_t>(written), stdout);
        std::fputs("\"buffers_cleared\":true,\"scope\":\"visual_classification_only\"}\n", stdout);
        local_tasks::secure_zero(result, sizeof(result));
        local_tasks::secure_zero(&task_id, sizeof(task_id));
        local_tasks::secure_zero(&predicted, sizeof(predicted));
        if (std::ferror(stdout)) return 5;
    }
    clear_job();
    local_tasks::secure_zero(result, sizeof(result));
    return 0;
}
