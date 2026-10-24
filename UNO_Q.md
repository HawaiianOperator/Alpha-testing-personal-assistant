# Arduino UNO Q 4 GB target

This is the current hardware profile. It supersedes ESP32-WROVER and UNO R3/R4.

## Separate processors and memory

Official Arduino specifications list a Qualcomm Dragonwing QRB2210 with four Cortex-A53 cores at 2.0 GHz, 4 GB LPDDR4 RAM, and 32 GB eMMC for Linux. The separate STM32U585 Cortex-M33 has up to 160 MHz, 786 KB SRAM, and 2 MB flash. Linux supports Docker/Compose; the MCU runs Arduino/Zephyr.

The 4 GB belongs to Linux, not the MCU. Track OS/services, model runtime, queue, and observation buffers on Linux separately from MCU firmware, stacks, SDK, communication, and control state.

The 1 GB model/storage project cap remains separate from these physical capacities.

## Proposed division

    Computer companion: selected-region capture and task entry
              -> local device connection
    UNO Q Linux: visual/language inference, clarification, RAM queue
              -> Arduino Bridge RPC
    UNO Q MCU: bounded control, watchdog, and state
              -> selected and tested host-input path
    Computer: input action and fresh observation

The official Bridge supports Linux-to-MCU RPC. This package does not yet implement or test a Bridge transport.

USB-C supports host/device role switching. The actual gadget/input endpoints and host connection must be verified; an ordinary MCU Keyboard sketch is not assumed to control the host. Linux USB gadget input, a compatible bridge, or native companion execution are integration options.

Camera ports do not read the computer's screen. The companion provides observations; leave unused cameras off.

## Current model and code

The trained small classifier can run on Linux. Its fixed 38,448-byte activation arena is well within the stated system RAM. A full MCU build may also be possible with the stated SRAM, but total SDK/stack memory and latency need target measurements.

Head-only deployment remains available: Linux runs the encoder and sends 16 INT8 features to the MCU. Its mathematical head payload is 84 bytes; pointers, alignment, stack, protocol, and SDK memory are extra.

The current model recognizes generated visual states. It is not the requested arbitrary-task language planner.

## Persistent classifier service

firmware/classifier_service.cpp keeps immutable weights loaded and accepts multiple binary requests through stdin/stdout.

Header: little-endian magic PAM1 (4 bytes), mode (1 byte), three reserved zero bytes, nonzero uint32 request ID, uint32 payload length. Mode1 payload is 4096 grayscale bytes; mode2 payload is 16 INT8 encoder features in [0,127].

It emits one classification response per frame and clears its owned arena, pixels, features, logits, header, and task context. It writes no task content to files and performs no desktop actions. This protocol is for a trusted local classifier process, not an authenticated network/USB/Bridge action service.

Native verification processed 32 requests in one process, checked exact reference logits and per-request buffer clearing, and rejected four malformed requests. Those checks were not run on UNO Q.

## Docker runtime

Build natively on the UNO Q's ARM64 Linux system:

    docker compose -f compose.uno-q.yaml build

The runtime recipe compiles the C++ service and needs no Python, CUDA, remote model download, or AI API during inference. Attach a separately implemented adapter to its binary input/output.

The sample container is read-only, has no network, disables Docker logging/core dumps, and limits memory and swap to 256 MB. This is an initial envelope for the tiny classifier, not a measured budget for a future language model. The future browser/calendar tool process needs its own explicitly scoped networking.

Docker image builds, ARM64 execution, Bridge, USB input, and hardware timing remain unverified here.

## From-scratch general model

The UNO Q provides a more suitable Linux execution target, but does not supply trained language/planning capability. A separate model needs an instruction/action corpus, autocomplete objective, evaluation, and deployment benchmarks. The included CUDA trainer trains the small vision architecture only.

Train in the user's Ampere virtual environment, export the needed components, then measure actual UNO Q memory and latency.

## Privacy

Keep active tasks, screen data, features, and clarification context in RAM. Retain immutable firmware/weights. Do not deliberately persist private task content to eMMC, logs, swap, crash dumps, or learning history.

Application-owned erasure does not cover unknown OS/driver/kernel copies. The board deployment still needs a logging/swap/core-dump audit.

Local AI does not make requested web searches or calendar operations offline: those tools intentionally communicate selected data to their providers. Disclose those transfers separately.

## Official references

- https://store.arduino.cc/products/uno-q-4gb
- https://docs.arduino.cc/hardware/uno-q/
- https://docs.arduino.cc/software/app-lab/bridge/bridge-api
- https://docs.arduino.cc/tutorials/uno-q/routerbridge-multilanguage
