import contextlib
import io
import json
import runpy
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from tests.unit_test import UnitTest


ROOT = Path(__file__).resolve().parent.parent


class Example:
    @staticmethod
    def identity(values):
        return values

    @staticmethod
    def total(values):
        return sum(values)


def write_suite(path, expected_total=3):
    suite = {
        "Example": {
            "identity": {
                "input": {"value": [1, 2], "format": "array"},
                "output": {"value": [1, 2], "format": "array"},
            },
            "total": {
                "input": {"value": [1, 2], "format": "array"},
                "output": {"value": expected_total, "format": "float"},
            },
        }
    }
    path.write_text(json.dumps(suite))


class UnitTestRunnerTests(unittest.TestCase):
    def test_check_all_returns_false_after_a_failed_assertion(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tests.json"
            write_suite(path, expected_total=4)
            output = io.StringIO()

            with contextlib.redirect_stdout(output):
                result = UnitTest.check_all(path, Example())

            self.assertFalse(result)
            self.assertIn("Example::identity", output.getvalue())
            self.assertIn("Example::total", output.getvalue())
            self.assertIn("--> OK", output.getvalue())
            self.assertIn("--> NOT OK", output.getvalue())

    def test_check_all_preserves_success_output(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tests.json"
            write_suite(path)
            output = io.StringIO()

            with contextlib.redirect_stdout(output):
                result = UnitTest.check_all(path, Example())

            self.assertTrue(result)
            self.assertEqual(output.getvalue().count("--> OK"), 2)
            self.assertNotIn("NOT OK", output.getvalue())

    def run_command_line_runner(self, suite_results):
        dataset_module = types.ModuleType("asalix.dataset_extractor")
        dataset_module.DatasetExtractor = type("DatasetExtractor", (), {})
        numerical_module = types.ModuleType("asalix.numerical_methods")
        numerical_module.NumericalMethods = type("NumericalMethods", (), {})
        fake_modules = {
            "asalix.dataset_extractor": dataset_module,
            "asalix.numerical_methods": numerical_module,
        }

        with mock.patch.dict(sys.modules, fake_modules):
            with mock.patch.object(UnitTest, "check_all", side_effect=suite_results) as check_all:
                with self.assertRaises(SystemExit) as raised:
                    runpy.run_path(str(ROOT / "run_tests.py"), run_name="__main__")

        self.assertEqual(check_all.call_count, 2)
        return raised.exception.code

    def test_command_line_runner_exits_non_zero_when_a_suite_fails(self):
        self.assertEqual(self.run_command_line_runner([True, False]), 1)

    def test_command_line_runner_exits_zero_when_all_suites_pass(self):
        self.assertEqual(self.run_command_line_runner([True, True]), 0)


if __name__ == "__main__":
    unittest.main()
