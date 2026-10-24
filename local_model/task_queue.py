"""Bounded, RAM-only task queue; owned task buffers are overwritten after use.

Python/OS copies outside these owned bytearrays are not claimed to be erased.
"""
from __future__ import annotations


def wipe(buffer):
    for index in range(len(buffer)):
        buffer[index] = 0


class VolatileTaskQueue:
    def __init__(self, capacity=8, max_request_bytes=256):
        if type(capacity) is not int or type(max_request_bytes) is not int or min(capacity, max_request_bytes) < 1:
            raise ValueError("Queue capacity and request length must be positive integers.")
        self.capacity = capacity
        self.max_request_bytes = max_request_bytes
        self._slots = [bytearray(max_request_bytes) for _ in range(capacity)]
        self._lengths = [0] * capacity
        self._ids = [0] * capacity
        self._active = bytearray(max_request_bytes)
        self._active_length = 0
        self.active_id = 0
        self._head = self._tail = self._count = 0
        self._next_id = 1
        self.closed = False

    @property
    def pending_count(self):
        return self._count

    @property
    def owned_request_buffer_bytes(self):
        return (self.capacity + 1) * self.max_request_bytes

    def enqueue(self, request):
        if self.closed:
            raise RuntimeError("Queue is closed.")
        if not isinstance(request, (bytes, bytearray, memoryview)):
            raise TypeError("Use bytes, bytearray, or a byte-format memoryview.")
        view = memoryview(request).cast("B")
        if not 1 <= len(view) <= self.max_request_bytes:
            raise ValueError("Request exceeds the fixed length or is empty.")
        if self._count == self.capacity:
            raise BufferError("Task queue is full.")
        task_id = self._next_id
        self._next_id += 1
        slot = self._slots[self._tail]
        slot[:len(view)] = view
        self._lengths[self._tail] = len(view)
        self._ids[self._tail] = task_id
        self._tail = (self._tail + 1) % self.capacity
        self._count += 1
        return task_id

    def start_next(self):
        if self.closed:
            raise RuntimeError("Queue is closed.")
        if self.active_id:
            raise RuntimeError("Finish the active task before starting another.")
        if not self._count:
            return None
        length = self._lengths[self._head]
        slot = self._slots[self._head]
        self._active[:length] = memoryview(slot)[:length]
        self._active_length = length
        self.active_id = self._ids[self._head]
        wipe(slot)
        self._lengths[self._head] = self._ids[self._head] = 0
        self._head = (self._head + 1) % self.capacity
        self._count -= 1
        return self.active_id, memoryview(self._active)[:length]

    def finish_active(self):
        wipe(self._active)
        self._active_length = 0
        self.active_id = 0

    def execute_next(self, handler):
        task = self.start_next()
        if task is None:
            return False
        task_id, view = task
        try:
            handler(task_id, view)
        finally:
            self.finish_active()
        return True

    def close(self):
        self.finish_active()
        for slot in self._slots:
            wipe(slot)
        self._lengths[:] = [0] * self.capacity
        self._ids[:] = [0] * self.capacity
        self._head = self._tail = self._count = 0
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
