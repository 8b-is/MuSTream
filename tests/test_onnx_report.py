"""Numerical export-check failures without model imports or downloads."""
import ast
import contextlib
import io
from pathlib import Path
import unittest

import numpy as np

SOURCE = Path(__file__).resolve().parents[1] / 'mustream/onnx_export/export_mert_onnx.py'
tree = ast.parse(SOURCE.read_text())
definitions = [node for node in tree.body
               if isinstance(node, ast.FunctionDef) and node.name == '_report']
if len(definitions) != 1:
    raise RuntimeError('Expected the actual shared export-check function')
namespace = {'np': np}
exec(compile(ast.Module(body=definitions, type_ignores=[]), str(SOURCE), 'exec'), namespace)
report = namespace['_report']


class ExportReportTests(unittest.TestCase):
    def check(self, actual, reference, tolerance=1e-4):
        with contextlib.redirect_stdout(io.StringIO()):
            return report('synthetic export', actual, reference, tolerance)

    def test_identical_and_scaled_finite_arrays_pass(self):
        self.assertEqual(self.check([0, 0], [0, 0]), 0)
        self.assertAlmostEqual(self.check([1001., -999.], [1000., -1000.], .01), .001)

    def test_exact_threshold_and_larger_error_fail(self):
        for actual in ([1.5], [2.]):
            with self.subTest(actual=actual), self.assertRaises(AssertionError):
                self.check(actual, [1.], .5)

    def test_nonfinite_actual_or_reference_fails(self):
        for value in (np.nan, np.inf, -np.inf):
            for actual, reference in (([value], [1.]), ([1.], [value]), ([value], [value])):
                with self.subTest(actual=actual, reference=reference):
                    with self.assertRaisesRegex(AssertionError, 'synthetic export.*finite'):
                        self.check(actual, reference)

    def test_broadcastable_shape_mismatch_fails(self):
        for actual, reference in (([[1., 2.]], [1., 2.]), (1., [1.]), ([[1.], [1.]], [[1., 1.]])):
            with self.subTest(actual=actual, reference=reference):
                with self.assertRaisesRegex(AssertionError, 'shape'):
                    self.check(actual, reference)

    def test_empty_comparison_has_a_clear_failure(self):
        with self.assertRaisesRegex(AssertionError, 'empty'):
            self.check([], [])

    def test_invalid_tolerances_fail_even_for_identical_arrays(self):
        for tolerance in (0., -1., np.nan, np.inf, -np.inf):
            with self.subTest(tolerance=tolerance):
                with self.assertRaisesRegex(ValueError, 'tolerance'):
                    self.check([1.], [1.], tolerance)

    def test_finite_inputs_with_overflowing_difference_fail(self):
        largest = np.finfo(np.float64).max
        with self.assertRaises(AssertionError):
            self.check([largest], [-largest])


if __name__ == '__main__':
    unittest.main()
