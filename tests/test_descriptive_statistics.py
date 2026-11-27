from math import sqrt

import pytest

from asalix import calculate_mean_value, calculate_standard_deviation


def test_arithmetic_mean():
    assert calculate_mean_value([1, 2, 3, 4]) == 2.5


@pytest.mark.parametrize('population, expected', [
    (True, sqrt(5 / 4)),
    (False, sqrt(5 / 3)),
], ids=['population', 'sample'])
def test_standard_deviation(population, expected):
    result = calculate_standard_deviation([1, 2, 3, 4], population=population)

    assert result == pytest.approx(expected, abs=1e-12)
