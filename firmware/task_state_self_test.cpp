#include "task_state.h"
#include <cstdio>
#include <limits>
namespace {
bool zeroed(const void* data, std::size_t size) {
    const auto* bytes = static_cast<const std::uint8_t*>(data);
    for (std::size_t i = 0; i < size; ++i) if (bytes[i]) return false;
    return true;
}
}
int main() {
    local_tasks::State state{};
    std::uint8_t workspace[4096];
    for (auto& byte : workspace) byte = 0xA5;
    const std::uint8_t request[] = {'p','r','i','v','a','t','e'};
    if (!local_tasks::begin(state, 7, request, sizeof(request))) return 1;
    if (local_tasks::begin(state, 8, request, sizeof(request))) return 2;
    state.features[0] = 127;
    state.logits[2] = -100;
    local_tasks::finish(state, workspace, sizeof(workspace));
    if (!zeroed(&state, sizeof(state)) || !zeroed(workspace, sizeof(workspace))) return 3;
    if (local_tasks::begin(state, 0, request, sizeof(request))) return 4;
    if (local_tasks::begin(state, 1, request, 257)) return 5;
    if (!local_tasks::begin(state, 8, request, 1)) return 6;
    if (state.request[1] != 0 || state.observation_id != 0 || state.action_index != 0) return 7;
    local_tasks::finish(state, nullptr, 0);
    if (!zeroed(&state, sizeof(state))) return 8;
    std::printf("{\"task_state_bytes\":%zu,\"wipe_verified\":true}\n", sizeof(state));
    return 0;
}
