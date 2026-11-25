from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import numpy as np
from scipy import stats

import asalix
from asalix.numerical_methods import NumericalMethods


class AndersonDarlingProbabilityTest(TestCase):
    def test_well_fitting_normal_sample_has_a_valid_probability(self):
        sample = stats.norm.ppf((np.arange(100) + 0.5) / 100)
        probability = asalix.normality_test(sample, test='anderson-darling')
        self.assertGreater(probability, 0.99)
        self.assertLessEqual(probability, 1)

    def test_small_adjusted_statistics_match_published_approximation(self):
        # Stephens' normality approximation, also used by statsmodels normal_ad:
        # https://www.statsmodels.org/stable/_modules/statsmodels/stats/_adnorm.html
        expected_values = [(0.1, 0.996148528516), (0.2, 0.884351516126)]
        correction = 1 + 0.75 / 100 + 2.25 / (100 * 100)
        for adjusted, expected in expected_values:
            with self.subTest(adjusted=adjusted), patch(
                    'asalix.numerical_methods.stats.anderson',
                    return_value=SimpleNamespace(statistic=adjusted / correction)):
                self.assertAlmostEqual(
                    NumericalMethods.anderson_darling_normality_test([0] * 100),
                    expected, places=10)

    def test_non_normal_sample_still_has_a_small_probability(self):
        sample = np.concatenate((np.zeros(50), np.ones(50)))
        probability = asalix.normality_test(sample, test='anderson-darling')
        self.assertGreaterEqual(probability, 0)
        self.assertLess(probability, 0.05)
