import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy import stats

import asalix
from asalix.plotter import Plotter


@pytest.fixture
def samples():
    return stats.norm.ppf(np.linspace(0.005, 0.995, 200)).reshape(20, 10)


@pytest.mark.parametrize('as_list', [False, True])
def test_plotted_histogram_matches_the_flattened_counts(monkeypatch, samples, as_list):
    monkeypatch.setattr(Plotter, 'show', staticmethod(lambda fig, ax: (fig, ax)))
    data = samples.tolist() if as_list else samples
    figure = 8301
    try:
        result = asalix.create_histogram(data, bins=8, plot=True, fig_number=figure)
        expected, _ = np.histogram(samples.ravel(), bins=8)
        np.testing.assert_array_equal(result.hist, expected)
        heights = [patch.get_height() for patch in plt.figure(figure).axes[0].patches]
        np.testing.assert_array_equal(heights, expected)
    finally:
        plt.close(figure)


@pytest.mark.parametrize('as_list', [False, True])
def test_quartile_plot_accepts_the_documented_flattened_input(monkeypatch, samples, as_list):
    monkeypatch.setattr(Plotter, 'show', staticmethod(lambda fig, ax: (fig, ax)))
    data = samples.tolist() if as_list else samples
    figure = 8302
    try:
        result = asalix.create_quartiles(data, plot=True, fig_number=figure)
        np.testing.assert_allclose([result.first, result.median, result.third], np.percentile(samples, [25, 50, 75]))
        assert len(plt.figure(figure).axes) == 1
        assert plt.figure(figure).axes[0].lines
    finally:
        plt.close(figure)


@pytest.mark.parametrize('as_list', [False, True])
def test_fitted_histogram_reports_one_normality_probability(samples, as_list):
    data = samples.tolist() if as_list else samples
    result = asalix.normal_distribution_fit(data)
    expected = asalix.normal_distribution_fit(samples.ravel())
    assert np.isscalar(result.p_value)
    np.testing.assert_allclose([result.p_value, result.normal_coefficient, result.mean_value, result.standard_deviation], [expected.p_value, expected.normal_coefficient, expected.mean_value, expected.standard_deviation])
