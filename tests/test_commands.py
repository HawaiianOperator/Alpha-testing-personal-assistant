import unittest
from local_model.commands import compute, route, suggestions, KeywordSession


class CommandTests(unittest.TestCase):
    def test_google_and_search_share_query(self):
        first = route("Google Nvidia")
        second = route("Search Nvidia")
        self.assertEqual(first["url"], second["url"])
        self.assertEqual(first["query"], "Nvidia")
        self.assertEqual(first["status"], "ready")

    def test_query_is_encoded_and_preserved(self):
        result = route("SEARCH: highest performance GPU & lowest cost")
        self.assertEqual(result["query"], "highest performance GPU & lowest cost")
        self.assertIn("%26", result["url"])
        self.assertEqual(route("google").get("status"), "needs_clarification")

    def test_compute_exact_examples(self):
        examples = {"2 + 2": "4", "0.1 + 0.2": "0.3",
                    "(2 + 3) * 4": "20", "-3 / 2": "-1.5", "1 - 1": "0"}
        for expression, value in examples.items():
            self.assertEqual(compute(expression), value)
        self.assertEqual(route("Compute 2 + 2")["value"], "4")

    def test_calls_variables_strings_and_powers_are_not_executed(self):
        for expression in ("__import__('os')", "open('secret')", "x + 1",
                           "'abc'", "True + 1", "2 ** 1000000", "[1][0]", "1e-999999999", "1e999999999"):
            with self.assertRaises(ValueError):
                compute(expression)
        self.assertEqual(route("Compute 1 / 0")["status"], "rejected")

    def test_boundaries(self):
        with self.assertRaises(ValueError):
            compute("1+" * 100 + "1")
        self.assertEqual(route("Google " + "x" * 300)["status"], "rejected")
        self.assertEqual(route("Googlesomething Nvidia")["status"], "rejected")

    def test_unsupported_tasks_reject_and_clear_context(self):
        message = "I am sorry the task is too complicated, try an less complicated task"
        for task in ("Optimize my calendar", "Compare GPUs and buy the best one", "Compute x + 1"):
            result = route(task)
            self.assertEqual(result["status"], "rejected")
            self.assertEqual(result["message"], message)
            self.assertNotIn("url", result)
        session = KeywordSession()
        session.accept("Compute")
        self.assertEqual(session.accept("x + 1")["status"], "rejected")
        self.assertIsNone(session.pending_command)
        self.assertEqual(session.accept("Compute 2 + 2")["value"], "4")
        self.assertEqual(route("")["status"], "needs_clarification")

    def test_keyword_autofill_has_no_history(self):
        self.assertEqual(suggestions("g"), ["google "])
        self.assertEqual(suggestions("comp"), ["compute "])
        self.assertEqual(suggestions("Google Nvidia"), [])

    def test_clarification_is_limited_to_current_task(self):
        session = KeywordSession()
        self.assertEqual(session.accept("Google")["command"], "search")
        self.assertEqual(session.accept("Nvidia")["query"], "Nvidia")
        self.assertIsNone(session.pending_command)
        self.assertEqual(session.accept("Compute")["status"], "needs_clarification")
        self.assertEqual(session.accept("2 + 2")["value"], "4")
        self.assertIsNone(session.pending_command)
        session.accept("Google")
        session.cancel()
        self.assertIsNone(session.pending_command)


if __name__ == "__main__":
    unittest.main()
