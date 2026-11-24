import numpy as np
import pytest
from scipy import stats

import asalix


@pytest.mark.parametrize('population', [False, True])
@pytest.mark.parametrize('shape', [(1, 6), (2, 3), (2, 1, 3)])
def test_confidence_interval_uses_every_observation(population, shape):
    flat = np.array([2.0, 4.0, 5.0, 7.0, 8.0, 10.0])
    data = flat.reshape(shape)
    coefficient = stats.norm.ppf(0.975) if population else stats.t.ppf(0.975, df=5)
    half_width = coefficient * flat.std(ddof=0 if population else 1) / np.sqrt(6)
    expected = (6.0 - half_width, 6.0 + half_width)
    np.testing.assert_allclose(asalix.calculate_confidence_interval(data, 0.95, population), expected)
    np.testing.assert_allclose(asalix.calculate_confidence_interval(data.tolist(), 0.95, population), expected)


@pytest.mark.parametrize('population', [False, True])
def test_one_dimensional_interval_is_unchanged(population):
    data = [2.0, 4.0, 6.0]
    coefficient = stats.norm.ppf(0.95) if population else stats.t.ppf(0.95, df=2)
    half_width = coefficient * np.std(data, ddof=0 if population else 1) / np.sqrt(3)
    np.testing.assert_allclose(asalix.calculate_confidence_interval(data, 0.9, population), (4 - half_width, 4 + half_width))
