from unittest import TestCase
from unittest.mock import patch

import asalix
import numpy as np
from asalix.numerical_methods import NelsonRule, NumericalMethods


class NelsonDispatchTest(TestCase):
    def test_rule_seven_and_eight_results_are_independent(self):
        # The first 15 points cluster near the mean; there is no run of
        # eight points outside one standard deviation on both sides.
        result = asalix.check_dataset_using_nelson_rules(np.array([0] * 15 + [10]))
        self.assertTrue(result.rule7)
        self.assertFalse(result.rule8)

    def test_each_rule_is_evaluated_once(self):
        with patch.object(NumericalMethods, 'perform_nelson_rule_test',
                          side_effect=lambda data, rule: rule.value) as evaluate:
            result = asalix.check_dataset_using_nelson_rules([1, 2, 3])
        self.assertEqual([call.args[1] for call in evaluate.call_args_list],
                         list(NelsonRule))
        self.assertEqual([getattr(result, 'rule' + str(n)) for n in range(1, 9)],
                         list(range(1, 9)))
