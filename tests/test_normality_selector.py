import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy import stats

import asalix

SAMPLE = np.linspace(0, 2, 40) ** 2
ANDERSON_DARLING_P_VALUE = 0.0012914573409241689


@pytest.mark.parametrize("name", ["anderson_darling", "anderson-darling"])
def test_anderson_darling_selector(name):
    # This sample is rejected by Anderson-Darling at 5%, but not by the default
    # D'Agostino-Pearson test (p = 0.07391273857867776).
    result = asalix.normality_test(SAMPLE, test=name)
    assert result == pytest.approx(ANDERSON_DARLING_P_VALUE)


@pytest.mark.parametrize("name", ["anderson_darling", "anderson-darling"])
def test_fit_forwards_anderson_darling_selector(name):
    sample = np.r_[stats.norm.ppf((np.arange(100) + 0.5) / 100), 5.0]
    result = asalix.normal_distribution_fit(sample, normal_distribution_test=name)
    assert result.p_value == pytest.approx(0.5638752678951622)


@pytest.mark.parametrize("name", ["anderson_darling", "anderson-darling"])
def test_histogram_shows_selected_normality_test(name, monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    number = 501
    try:
        asalix.create_histogram(
            SAMPLE, normal_distribution_test=name, plot=True, fig_number=number
        )
        axis = plt.figure(number).axes[0]
        text = "\n".join(artist.txt.get_text() for artist in axis.artists)
        assert "p-value:  0.001" in text
    finally:
        plt.close(number)


def test_other_normality_selectors_keep_their_results():
    assert asalix.normality_test(SAMPLE) == pytest.approx(
        stats.normaltest(SAMPLE).pvalue
    )
    assert asalix.normality_test(SAMPLE, test="basic") == pytest.approx(
        stats.normaltest(SAMPLE).pvalue
    )
    assert asalix.normality_test(SAMPLE, test="shapiro_wilk") == pytest.approx(
        stats.shapiro(SAMPLE).pvalue
    )
    standardized = (SAMPLE - SAMPLE.mean()) / SAMPLE.std()
    assert asalix.normality_test(SAMPLE, test="kolmogorov_smirnov") == pytest.approx(
        stats.kstest(standardized, "norm").pvalue
    )
