import json
from pathlib import Path
import subprocess
import sys
import unittest
from local_model.text_assistant import IntentModel, Session, handle
from local_model.text_features import DIMENSIONS, features
from local_model.commands import REJECTION_MESSAGE

ROOT=Path(__file__).resolve().parent.parent

class TextAssistantTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.model=IntentModel(ROOT/"artifacts_intent/intent_model.json")
    def test_numeric_arguments_are_computed_not_memorized(self):
        for task,value in (("Compute 2 + 2","4"),("Calculate 1729 + 2831","4560"),
            ("What is 7 * 13?","91"),("Please evaluate 0.25 + 0.75","1")):
            self.assertEqual(handle(task,self.model).get("value"),value)
    def test_search_preserves_argument_without_network(self):
        for task in ("Search Nvidia","Google Nvidia","Look up Nvidia on Google"):
            result=handle(task,self.model)
            self.assertEqual(result.get("query"),"Nvidia")
            self.assertEqual(result["status"],"ready")
            self.assertEqual(result["url"],"https://www.google.com/search?q=Nvidia")
    def test_complex_tasks_reject_exactly(self):
        for task in ("generate an image of the empire state building", "Optimize my calendar",
            "Search GPUs then buy one", "Compute 2+2 and email it", "Search GPUs and install drivers"):
            self.assertEqual(handle(task,self.model),{"status":"rejected","message":REJECTION_MESSAGE})
    def test_unsupported_arithmetic_cannot_execute_code(self):
        for task in ("Compute __import__('os')", "Calculate 2 ** 99999", "Compute 1 / 0", "Compute x + 1"):
            self.assertEqual(handle(task,self.model)["status"],"rejected")
    def test_clarification_does_not_persist_completed_context(self):
        session=Session(self.model)
        self.assertEqual(session.accept("Compute")["status"],"needs_clarification")
        self.assertEqual(session.accept("2 + 2")["value"],"4")
        self.assertIsNone(session.pending)
        self.assertEqual(session.accept("Search")["status"],"needs_clarification")
        self.assertEqual(session.accept("Nvidia")["query"],"Nvidia")
        self.assertIsNone(session.pending)
        session.accept("Compute"); session.accept("cancel")
        self.assertIsNone(session.pending)
        self.assertEqual(session.accept("generate an image")["status"],"rejected")
        self.assertIsNone(session.pending)
    def test_bounds_and_feature_contract(self):
        self.assertEqual(handle("Search "+"x"*300,self.model)["status"],"rejected")
        self.assertEqual(handle("",self.model)["status"],"needs_clarification")
        self.assertEqual(features("Compute 2 + 2"),features("Compute 102 + 376"))
        self.assertTrue(all(0<=k<DIMENSIONS for k in features("Nvidia")))
        self.assertAlmostEqual(sum(v*v for v in features("Compute 2 + 2").values()),1)
    def test_runtime_without_site_packages_and_jsonl_protocol(self):
        result=subprocess.run([sys.executable,"-S","-m","local_model.text_assistant","--jsonl"],
            cwd=ROOT,input="Compute 2 + 2\ngenerate an image of the empire state building\nexit\n",
            text=True,capture_output=True,check=True)
        rows=[json.loads(line) for line in result.stdout.splitlines()]
        self.assertEqual(len(rows),2)
        self.assertEqual(rows[0]["value"],"4")
        self.assertEqual(rows[1]["message"],REJECTION_MESSAGE)

if __name__ == "__main__": unittest.main()
