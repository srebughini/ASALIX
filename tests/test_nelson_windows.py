from unittest import TestCase
from unittest.mock import patch

from asalix.numerical_methods import NumericalMethods


class NelsonWindowsTest(TestCase):
    def test_rule_seven_checks_final_window(self):
        self.assertTrue(NumericalMethods.perform_nelson_rule_7_test([100] + [0] * 15))

    def test_rule_eight_checks_final_window(self):
        self.assertTrue(NumericalMethods.perform_nelson_rule_8_test([0] * 20 + [-3, 3] * 4))

    def test_rule_eight_requires_both_sides(self):
        for value in (3, -3):
            with self.subTest(value=value):
                self.assertFalse(NumericalMethods.perform_nelson_rule_8_test([0] * 20 + [value] * 9))

    def test_exact_window_sizes(self):
        # Fixed control limits isolate window enumeration from the estimation
        # of mean/sigma using the same short sample being tested.
        with patch.object(NumericalMethods, 'calculate_mean_value', return_value=0), \
                patch.object(NumericalMethods, 'calculate_sample_standard_deviation', return_value=1):
            self.assertTrue(NumericalMethods.perform_nelson_rule_7_test([0] * 15))
            self.assertFalse(NumericalMethods.perform_nelson_rule_7_test([0] * 14))
            self.assertTrue(NumericalMethods.perform_nelson_rule_8_test([-2, 2] * 4))
            self.assertFalse(NumericalMethods.perform_nelson_rule_8_test([-2, 2] * 3 + [2]))
            self.assertFalse(NumericalMethods.perform_nelson_rule_8_test([2] * 8))
            self.assertFalse(NumericalMethods.perform_nelson_rule_8_test([-2, 2] * 3 + [0, 2]))
