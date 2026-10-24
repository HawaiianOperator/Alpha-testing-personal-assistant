// Board-agnostic entry point. Wire it to the ESP-IDF observation receiver.
// This does not initialize UART, Bluetooth HID, or a task scheduler.
#include "model_runtime.h"
#include "model_data.h"

namespace {
local_model::Arena arena;
}

// The input frame belongs to the caller and must remain valid for the call.
// Do not call concurrently: this wrapper deliberately shares one static arena.
extern "C" bool model_process_observation(const std::uint8_t* grayscale,
                                           std::size_t byte_count,
                                           std::int8_t* output_logits,
                                           std::uint8_t* output_class) {
    if (byte_count != local_model::kInputBytes || !output_class) {
        return false;
    }
    const int predicted =
        local_model::infer(model_data::kModel, arena, grayscale, output_logits);
    if (predicted < 0) {
        return false;
    }
    *output_class = static_cast<std::uint8_t>(predicted);
    return true;
}
