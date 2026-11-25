import numpy as np
import pytest

import asalix
from asalix.numerical_methods import NumericalMethods


def test_constant_process_has_no_three_sigma_outlier():
    assert not NumericalMethods.perform_nelson_rule_1_test(np.full(30, 12.5))
    assert not asalix.check_dataset_using_nelson_rules(np.full(30, 12.5)).rule1


@pytest.mark.parametrize('boundary', [-3.0, 3.0])
def test_point_exactly_on_control_limit_is_not_beyond_it(monkeypatch, boundary):
    monkeypatch.setattr(NumericalMethods, 'calculate_mean_value', staticmethod(lambda data: 0.0))
    monkeypatch.setattr(NumericalMethods, 'calculate_sample_standard_deviation', staticmethod(lambda data: 1.0))
    assert not NumericalMethods.perform_nelson_rule_1_test([0.0, boundary, 0.0])


@pytest.mark.parametrize('outlier', [-3.01, 3.01])
def test_point_beyond_control_limit_is_detected(monkeypatch, outlier):
    monkeypatch.setattr(NumericalMethods, 'calculate_mean_value', staticmethod(lambda data: 0.0))
    monkeypatch.setattr(NumericalMethods, 'calculate_sample_standard_deviation', staticmethod(lambda data: 1.0))
    assert NumericalMethods.perform_nelson_rule_1_test([0.0, outlier, 0.0])


def test_real_extreme_reading_is_still_detected():
    assert NumericalMethods.perform_nelson_rule_1_test([0.0] * 30 + [100.0])
