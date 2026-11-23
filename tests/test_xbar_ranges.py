import unittest
from unittest.mock import patch

from asalix.numerical_methods import NumericalMethods, ControlChartTablesCoefficient


class XbarRangeTests(unittest.TestCase):
    def test_first_subgroup_contributes_to_average_range(self):
        dataset = {"first": [0, 3, 6], "second": [2, 3, 4], "third": [2, 3, 4]}
        limits = NumericalMethods.calculate_control_limits(dataset, "XbarR")
        self.assertAlmostEqual(limits[5], 10 / 3)
        reordered = {key: dataset[key] for key in ("second", "third", "first")}
        self.assertEqual(limits, NumericalMethods.calculate_control_limits(reordered, "XbarR"))

    def test_limits_use_the_average_of_all_subgroup_ranges(self):
        dataset = {"first": [0, 3, 6], "second": [2, 3, 4], "third": [2, 3, 4]}
        coefficients = {
            ControlChartTablesCoefficient.A2: 1.023,
            ControlChartTablesCoefficient.D3: 0,
            ControlChartTablesCoefficient.D4: 2.574,
        }
        # Isolate range averaging from the separately tracked coefficient lookup bug.
        with patch.object(NumericalMethods, "_access_to_control_chart_tables_coefficient",
                          side_effect=lambda n, coefficient: coefficients[coefficient]):
            limits = NumericalMethods.calculate_control_limits(dataset, "XbarR")
        expected = (3 - 1.023 * 10 / 3, 0, 3 + 1.023 * 10 / 3, 2.574 * 10 / 3, 3, 10 / 3)
        for actual, value in zip(limits, expected):
            self.assertAlmostEqual(actual, value)


if __name__ == "__main__":
    unittest.main()
