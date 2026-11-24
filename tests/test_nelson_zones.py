from unittest import TestCase
from unittest.mock import patch

from asalix.numerical_methods import NumericalMethods


class NelsonZoneTest(TestCase):
    def setUp(self):
        mean = patch.object(NumericalMethods, 'calculate_mean_value', return_value=0)
        sigma = patch.object(NumericalMethods, 'calculate_sample_standard_deviation', return_value=1)
        mean.start()
        sigma.start()
        self.addCleanup(mean.stop)
        self.addCleanup(sigma.stop)

    def test_two_of_three_beyond_two_sigma(self):
        for sign in (1, -1):
            for values in ([3, 0, 3], [0, 3, 3], [3, 3, 0], [3, 3, 3]):
                with self.subTest(sign=sign, values=values):
                    self.assertTrue(NumericalMethods.perform_nelson_rule_5_test(
                        [sign * value for value in values]))

    def test_four_of_five_beyond_one_sigma(self):
        for sign in (1, -1):
            for inside in range(5):
                values = [2 * sign] * 5
                values[inside] = 0
                self.assertTrue(NumericalMethods.perform_nelson_rule_6_test(values))
            self.assertTrue(NumericalMethods.perform_nelson_rule_6_test([2 * sign] * 5))

    def test_final_windows_are_checked(self):
        self.assertTrue(NumericalMethods.perform_nelson_rule_5_test([0, 0, 0, 3, 0, 3]))
        self.assertTrue(NumericalMethods.perform_nelson_rule_6_test([0, 0, 0, 2, 2, 0, 2, 2]))

    def test_thresholds_directions_and_separate_windows_do_not_combine(self):
        for values in ([], [3, 3], [3, 0, 0], [3, 0, -3], [2, 2, 2],
                       [3, 0, 0, 3, 0, 0, 3]):
            with self.subTest(rule=5, values=values):
                self.assertFalse(NumericalMethods.perform_nelson_rule_5_test(values))
        for values in ([], [2] * 4, [2, 2, 2, 0, 0], [2, 2, -2, -2, 0],
                       [1] * 5, [2, 2, 2, 0, 0, 2, 2, 2]):
            with self.subTest(rule=6, values=values):
                self.assertFalse(NumericalMethods.perform_nelson_rule_6_test(values))


class NelsonZoneRealDataTest(TestCase):
    def test_real_data_reaches_each_threshold(self):
        self.assertTrue(NumericalMethods.perform_nelson_rule_5_test([0] * 30 + [10, 0, 10]))
        self.assertTrue(NumericalMethods.perform_nelson_rule_6_test([0] * 30 + [10, 10, 0, 10, 10]))
