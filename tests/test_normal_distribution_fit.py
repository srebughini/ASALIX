import numpy as np

from asalix.numerical_methods import NumericalMethods


def test_gaussian_values_on_both_sides_of_the_peak():
    values = NumericalMethods.calculate_normal_distribution([1, 2, 3, 4, 5], 3, 3, 1.5)

    np.testing.assert_allclose(values, [1.23333687, 2.40221221, 3.0, 2.40221221, 1.23333687])


def test_sparse_histogram_fit_is_nearly_flat_at_the_mean_count():
    counts = [1, 0, 0, 1, 0, 0, 1, 0, 0, 1]
    centres = [1.15, 1.45, 1.75, 2.05, 2.35, 2.65, 2.95, 3.25, 3.55, 3.85]
    amplitude, mean, sigma = NumericalMethods.normal_distribution_fit(counts, centres, 1.0, 1.0, 1.0)

    # Many centres and widths give the same nearly constant curve on these bins.
    assert np.isfinite([amplitude, mean, sigma]).all()
    assert sigma > 0
    fitted = NumericalMethods.calculate_normal_distribution(centres, amplitude, mean, sigma)
    np.testing.assert_allclose(fitted, np.mean(counts), rtol=0, atol=1e-3)


def test_fit_recovers_parameters_of_a_known_gaussian():
    centres = [1, 2, 3, 4, 5]
    counts = [1, 8, 16, 8, 1]
    # The peak is at 3 and the half-height points are one unit away.
    sigma = 1 / np.sqrt(2 * np.log(2))
    fitted = NumericalMethods.normal_distribution_fit(counts, centres, 12.0, 2.5, 1.0)

    np.testing.assert_allclose(fitted, [16.0, 3.0, sigma], rtol=1e-6)
