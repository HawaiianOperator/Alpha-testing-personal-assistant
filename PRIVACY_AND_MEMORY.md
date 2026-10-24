# Active-state privacy and memory contract

## Persistent code and private task data

Firmware and immutable model weights remain stored or must be provisioned again after boot. The active privacy requirement is interpreted as **no persistent user-task content** on the board.

Requests, screen observations, feature vectors, action history, and clarification context must not be written to flash, EEPROM, SD, logs, a database, or crash records by the active application. Platform libraries must be audited before claiming this behavior for an actual board.

If literally no persistent data of any kind is allowed, firmware/model provisioning also needs a different design.

## One task at a time

The host owns the pending task queue in RAM. Only one job owns the controller at a time. Future queued requests remain in RAM until processed or cancelled; clearing a completed job must not clear the future queue.

Completion, errors, cancellation, timeout, and disconnect use the same cleanup path. Send the intended result, release held input through the future executor, overwrite all private buffers, then reset indices/state.

The neural network is feed-forward. It has no recurrent task cache and is not updated during use. Immutable weights do not contain the previous active request.

## Speed-first allocation

Keep immutable weights ready. Use fixed-size buffers and a fixed-capacity queue. Overwrite/reset mutable arenas after each job instead of repeatedly allocating and freeing them.

For head-only deployment:

    weights = 4 * 16 INT8 = 64 bytes
    biases = 4 INT32 = 16 bytes
    multiplier = 1 INT32 = 4 bytes
    mathematical payload = 84 bytes
    feature vector = 16 bytes
    output vector = 4 bytes

Pointers, structure alignment, call stack, serial/input libraries, task state, and the rest of firmware are additional. The exported Model/HeadModel descriptor already contains multipliers; do not count them twice.

The full CNN uses:

    A = ceil(7,688 / 16) * 16 = 7,696
    B = 30,752
    activation arena = 38,448 bytes

On UNO Q this arena can be a Linux-side component; a full STM32 build also requires separate total-memory and latency measurements. Linux 4 GB RAM and MCU 786 KB SRAM are separate budgets. A separate raw input image adds 4,096 bytes unless buffer ownership/layout avoids the copy.

The host queue's default request buffers are:

    (8 pending slots + 1 active slot) * 256 = 2,304 bytes

This is host memory, not UNO SRAM, and excludes Python objects/runtime overhead.

## Cleanup implementation

firmware/task_state.cpp uses volatile writes for explicit erasure of registered regions:

```cpp
local_tasks::Region private_buffers[] = {
    {&arena, sizeof(arena)},
    {pixels, sizeof(pixels)}
};
local_tasks::finish_all(task, private_buffers, 2);
```

Register every private buffer, including transport receive buffers and any separately retained outputs. Head-only deployment has no full arena to register, but still needs feature/request/transport cleanup.

Native tests check owned state and workspace bytes after cleanup. This does not establish erasure of unknown copies, registers, OS caches/swap, SDK internals, or physical remanence. The Python queue likewise overwrites owned bytearrays; callers can create copies outside that ownership boundary.

The current queue is single-threaded. A threaded UI/transport requires a separately designed synchronization boundary.

## Training versus active operation

The training pipeline deliberately stores labeled development data, checkpoints, and evaluation results. It must not silently collect active user tasks. Use synthetic or separately authorized/curated examples.

The supplied weights were trained on generated fixtures only. A CUDA trainer is provided for the user's Ampere environment but has not been run there.

## Product privacy statement

A supportable design goal is: **local inference, no external AI API dependency, no intentional active-task persistence, and explicit clearing of application-owned state**.

Web research intentionally sends the chosen query to a search provider. Calendar operations interact with and may persist information in the chosen calendar service. Disclose these transfers separately from AI inference. Do not promise that no data can ever leave the computer, or make unverified claims about competitors' data handling.

## Current verification boundary

Native-host model training, integer full/head inference, fixed activation layout, and owned-buffer tests are verified. Arduino toolchain/flash layout, actual input control, board stack usage, GPU/VM execution, Docker builds, and system-wide data erasure are not yet verified.
