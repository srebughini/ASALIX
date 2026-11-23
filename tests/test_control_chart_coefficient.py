import numpy as np

from asalix.numerical_methods import ControlChartTablesCoefficient, NumericalMethods


def test_coefficient_lookup_returns_coefficient_for_requested_sample_size():
    # The coefficient must correspond to the requested sample size,
    # not to the first key of the table.
    assert NumericalMethods._access_to_control_chart_tables_coefficient(
        4, ControlChartTablesCoefficient.B4) == 2.266
    assert NumericalMethods._access_to_control_chart_tables_coefficient(
        9, ControlChartTablesCoefficient.B4) == 1.761
    assert NumericalMethods._access_to_control_chart_tables_coefficient(
        15, ControlChartTablesCoefficient.B4) == 1.572
    assert NumericalMethods._access_to_control_chart_tables_coefficient(
        10, ControlChartTablesCoefficient.A2) == 0.308


def test_coefficient_lookup_nearest_sample_size_for_unsupported_sizes():
    # For sample sizes not present in the table, the closest available
    # sample size is used.
    coefficient = NumericalMethods._access_to_control_chart_tables_coefficient(
        11, ControlChartTablesCoefficient.B4)
    keys = np.asarray(list(ControlChartTablesCoefficient.B4.value.keys()))
    expected_key = keys[np.argmin(np.abs(keys - 11))]
    assert coefficient == ControlChartTablesCoefficient.B4.value[expected_key]
