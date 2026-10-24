import unittest
from local_model.task_queue import VolatileTaskQueue


class QueueTests(unittest.TestCase):
    def test_fifo_finish_wipes_active_but_keeps_pending(self):
        queue = VolatileTaskQueue(capacity=2, max_request_bytes=16)
        first = queue.enqueue(b"private first")
        second = queue.enqueue(b"second")
        task_id, view = queue.start_next()
        self.assertEqual(task_id, first)
        self.assertEqual(bytes(view), b"private first")
        with self.assertRaises(RuntimeError):
            queue.start_next()
        queue.finish_active()
        self.assertEqual(bytes(view), bytes(len(view)))
        self.assertTrue(all(value == 0 for value in queue._active))
        task_id, view = queue.start_next()
        self.assertEqual(task_id, second)
        self.assertEqual(bytes(view), b"second")
        queue.close()
        self.assertEqual(bytes(view), bytes(len(view)))
        self.assertTrue(all(not any(slot) for slot in queue._slots))

    def test_exception_still_wipes_task(self):
        queue = VolatileTaskQueue()
        queue.enqueue(b"secret")
        def fail(_, view):
            self.assertEqual(bytes(view), b"secret")
            raise ValueError("simulated action failure")
        with self.assertRaises(ValueError):
            queue.execute_next(fail)
        self.assertEqual(queue.active_id, 0)
        self.assertFalse(any(queue._active))

    def test_limits_and_cancellation(self):
        with VolatileTaskQueue(capacity=1, max_request_bytes=4) as queue:
            self.assertEqual(queue.owned_request_buffer_bytes, 8)
            with self.assertRaises(ValueError):
                queue.enqueue(b"large")
            with self.assertRaises(TypeError):
                queue.enqueue("text")
            queue.enqueue(b"one")
            with self.assertRaises(BufferError):
                queue.enqueue(b"two")
        self.assertEqual(queue.pending_count, 0)
        self.assertTrue(all(not any(slot) for slot in queue._slots))
        with self.assertRaises(RuntimeError):
            queue.enqueue(b"x")


if __name__ == "__main__":
    unittest.main()
