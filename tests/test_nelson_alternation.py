from unittest import TestCase

import numpy as np

from asalix.numerical_methods import NumericalMethods, NelsonRule


class NelsonAlternationTest(TestCase):
    def test_fourteen_alternating_points_in_either_direction(self):
        for values in ([0, 1] * 7, [1, 0] * 7):
            with self.subTest(values=values):
                self.assertTrue(NumericalMethods.perform_nelson_rule_test(
                    np.asarray(values), NelsonRule.RULE4))

    def test_rising_or_falling_step_sizes_are_not_alternation(self):
        for sign in (1, -1):
            values = sign * np.arange(30) ** 2
            self.assertFalse(NumericalMethods.perform_nelson_rule_4_test(values))

    def test_plateaus_and_repeated_directions_break_a_run(self):
        for values in ([0, 1] * 6 + [0], [0, 1] * 6 + [1, 0],
                       [0, 1] * 5 + [2, 1, 2, 1], [1] * 20):
            with self.subTest(values=values):
                self.assertFalse(NumericalMethods.perform_nelson_rule_4_test(np.asarray(values)))

    def test_final_window_and_list_input(self):
        self.assertTrue(NumericalMethods.perform_nelson_rule_4_test([0, 0] + [0, 2] * 7))

    def test_unsigned_integer_values_keep_their_direction(self):
        self.assertTrue(NumericalMethods.perform_nelson_rule_4_test(
            np.asarray([0, 1] * 7, dtype=np.uint8)))
