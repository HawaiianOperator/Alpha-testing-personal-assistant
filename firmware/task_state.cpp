#include "task_state.h"
namespace local_tasks {
void secure_zero(void* memory, std::size_t size) {
    if (!memory) return;
    volatile std::uint8_t* bytes = static_cast<volatile std::uint8_t*>(memory);
    for (std::size_t i = 0; i < size; ++i) bytes[i] = 0;
}
bool begin(State& state, std::uint32_t id, const std::uint8_t* request, std::size_t size) {
    if (state.active || !request || id == 0 || size == 0 || size > kRequestCapacity) return false;
    secure_zero(&state, sizeof(state));
    state.task_id = id;
    state.request_length = static_cast<std::uint16_t>(size);
    for (std::size_t i = 0; i < size; ++i) state.request[i] = request[i];
    state.active = 1;
    return true;
}
void finish_all(State& state, const Region* regions, std::size_t count) {
    if (regions) for (std::size_t i = 0; i < count; ++i) secure_zero(regions[i].data, regions[i].size);
    secure_zero(&state, sizeof(state));
}
void finish(State& state, void* inference_workspace, std::size_t workspace_bytes) {
    secure_zero(inference_workspace, workspace_bytes);
    secure_zero(&state, sizeof(state));
}
}
