from unittest import TestCase

import numpy as np

from asalix.numerical_methods import NumericalMethods, NelsonRule


class NelsonTrendTest(TestCase):
    def test_six_points_are_enough_for_either_direction(self):
        for values in ([1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1]):
            with self.subTest(values=values):
                self.assertTrue(NumericalMethods.perform_nelson_rule_test(
                    np.asarray(values), NelsonRule.RULE3))

    def test_short_runs_plateaus_and_direction_changes_reset_the_trend(self):
        for values in ([1, 2, 3, 4, 5], [1, 2, 3, 3, 4, 5, 6],
                       [1, 2, 3, 4, 3, 2, 1], [2] * 20):
            with self.subTest(values=values):
                self.assertFalse(NumericalMethods.perform_nelson_rule_3_test(np.asarray(values)))

    def test_final_run_and_list_input(self):
        self.assertTrue(NumericalMethods.perform_nelson_rule_3_test([9, 1, 2, 3, 4, 5, 6]))
        self.assertTrue(NumericalMethods.perform_nelson_rule_3_test([1, 6, 5, 4, 3, 2, 1]))

    def test_unsigned_decreases_do_not_wrap_into_an_increasing_trend(self):
        values = np.asarray([1, 2, 1, 2, 1, 2, 1, 2], dtype=np.uint8)
        self.assertFalse(NumericalMethods.perform_nelson_rule_3_test(values))
