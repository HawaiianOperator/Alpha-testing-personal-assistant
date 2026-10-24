#ifndef LOCAL_TASK_STATE_H
#define LOCAL_TASK_STATE_H
#include <cstddef>
#include <cstdint>

namespace local_tasks {
constexpr std::size_t kRequestCapacity = 256;

struct State {
    std::uint32_t task_id;
    std::uint32_t observation_id;
    std::uint32_t action_index;
    std::uint16_t request_length;
    std::uint8_t active;
    std::uint8_t stage;
    std::uint8_t request[kRequestCapacity];
    std::int8_t features[16];
    std::int8_t logits[4];
};

// Only buffers supplied to this API are covered by its clearing behavior.
struct Region { void* data; std::size_t size; };
void secure_zero(void* memory, std::size_t size);
void finish_all(State& state, const Region* regions, std::size_t count);
bool begin(State& state, std::uint32_t id, const std::uint8_t* request, std::size_t size);
void finish(State& state, void* inference_workspace, std::size_t workspace_bytes);
}
#endif
