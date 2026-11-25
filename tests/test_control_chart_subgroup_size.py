import math

import pytest

from asalix.numerical_methods import NumericalMethods


@pytest.mark.parametrize("chart", ["XbarR", "XbarS"])
@pytest.mark.parametrize("observations", [[1, 3, 5], [1, 2, 3, 4, 5]])
def test_repeating_subgroups_does_not_change_limits(chart, observations):
    few = {i: observations for i in range(2)}
    many = {i: observations for i in range(8)}
    assert NumericalMethods.calculate_control_limits(few, chart) == pytest.approx(
        NumericalMethods.calculate_control_limits(many, chart)
    )


@pytest.mark.parametrize("size, a2, d3, d4, a3, b3, b4", [
    (3, 1.023, 0, 2.574, 1.954, 0, 2.568),
    (5, 0.577, 0, 2.114, 1.427, 0, 2.089),
    (6, 0.483, 0, 2.004, 1.287, 0.03, 1.97),
])
def test_limits_use_coefficients_for_observations_per_subgroup(size, a2, d3, d4, a3, b3, b4):
    observations = list(range(1, size + 1))
    dataset = {i: observations for i in range(8)}
    mean = (size + 1) / 2
    spread = size - 1
    stdev = math.sqrt(sum((x - mean) ** 2 for x in observations) / (size - 1))
    assert NumericalMethods.calculate_control_limits(dataset, "XbarR") == pytest.approx(
        (mean - a2 * spread, d3 * spread, mean + a2 * spread, d4 * spread, mean, spread)
    )
    assert NumericalMethods.calculate_control_limits(dataset, "XbarS") == pytest.approx(
        (mean - a3 * stdev, b3 * stdev, mean + a3 * stdev, b4 * stdev, mean, stdev)
    )


@pytest.mark.parametrize("chart", ["XbarR", "XbarS"])
@pytest.mark.parametrize("dataset", [{}, {0: []}, {0: [1]}, {0: [1, 2], 1: [1, 2, 3]}])
def test_control_limits_require_equal_subgroups_with_at_least_two_observations(chart, dataset):
    with pytest.raises(ValueError, match="subgroup"):
        NumericalMethods.calculate_control_limits(dataset, chart)
