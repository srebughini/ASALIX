import warnings
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import stats

import asalix


@pytest.mark.parametrize("half_size", [500, 1000, 2000])
def test_large_bimodal_samples_have_a_small_finite_probability(half_size):
    sample = np.r_[
        np.linspace(-5.2, -4.8, half_size),
        np.linspace(4.8, 5.2, half_size),
    ]
    control = stats.anderson(sample)
    assert control.statistic > control.critical_values[-1]

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        probability = asalix.normality_test(sample, test="anderson-darling")

    assert np.isfinite(probability)
    assert 0 <= probability < 5e-31


@pytest.mark.parametrize(
    "adjusted, expected",
    [
        (1.0, 0.012317922048460179),
        (10.0, 3.764978805388397e-24),
        (13.0, 4.954210805845892e-31),
        (13.1, 0.0),
    ],
)
def test_tail_approximation_matches_reference_range(adjusted, expected, monkeypatch):
    # statsmodels' normal_ad scalar approximation uses the polynomial through
    # adjusted AD = 13, and reports zero above that range (p < 4.955e-31).
    # https://www.statsmodels.org/stable/_modules/statsmodels/stats/_adnorm.html
    sample = np.array([-1.0, 0.0, 1.0])
    correction = 1 + 0.75 / 3 + 2.25 / 9
    monkeypatch.setattr(
        "asalix.numerical_methods.stats.anderson",
        lambda dataset: SimpleNamespace(statistic=adjusted / correction),
    )
    probability = asalix.normality_test(sample, test="anderson-darling")
    assert probability == pytest.approx(expected, rel=1e-12, abs=0.0)


def test_ordinary_non_normal_sample_keeps_its_probability():
    sample = np.linspace(0, 2, 40) ** 2
    probability = asalix.normality_test(sample, test="anderson-darling")
    assert probability == pytest.approx(0.0012914573409241689)
