import numpy as np
import pytest

from asalix.numerical_methods import NumericalMethods


@pytest.mark.parametrize('edges, expected', [
    (np.array([120, 122, 124, 126], dtype=np.int8), [121.0, 123.0, 125.0]),
    (np.array([250, 252, 254], dtype=np.uint8), [251.0, 253.0]),
    (np.array([-126, -124, -122, -120], dtype=np.int8), [-125.0, -123.0, -121.0]),
    (np.array([1.8e38, 2.0e38, 2.2e38], dtype=np.float32), [1.9e38, 2.1e38]),
])
def test_bin_centres_do_not_overflow_the_edge_dtype(edges, expected):
    original = edges.copy()
    hist, returned_edges, centres = NumericalMethods.create_histogram(edges, bins=edges)
    np.testing.assert_allclose(centres, expected, rtol=1e-7)
    assert np.isfinite(centres).all()
    assert (centres >= edges[:-1]).all()
    assert (centres <= edges[1:]).all()
    np.testing.assert_array_equal(returned_edges, original)
    np.testing.assert_array_equal(edges, original)
    np.testing.assert_array_equal(hist, [1] * (len(edges) - 2) + [2])


def test_ordinary_nonuniform_edges_keep_their_counts_and_centres():
    hist, edges, centres = NumericalMethods.create_histogram([0, 1, 2, 3, 4], bins=[0, 1, 4])
    np.testing.assert_array_equal(hist, [1, 4])
    np.testing.assert_array_equal(edges, [0, 1, 4])
    np.testing.assert_allclose(centres, [0.5, 2.5])
