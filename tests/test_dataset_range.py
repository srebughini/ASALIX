import numpy as np
import pytest

from asalix.numerical_methods import NumericalMethods


@pytest.mark.parametrize('dataset', [
    [[1, 3], [2, 8]],
    np.array([[1, 3], [2, 8]]),
    np.array([1, 3, 2, 8]).reshape(1, 2, 2),
])
def test_range_uses_every_observation(dataset):
    assert NumericalMethods.calculate_range_value(dataset) == 7


@pytest.mark.parametrize('dtype', [np.int8, np.int16, np.int32, np.int64])
def test_signed_integer_range_does_not_wrap(dtype):
    limits = np.iinfo(dtype)
    dataset = np.array([limits.min, 0, limits.max], dtype=dtype)
    before = dataset.copy()
    expected = int(limits.max) - int(limits.min)
    with np.errstate(over='raise'):
        assert NumericalMethods.calculate_range_value(dataset) == expected
    np.testing.assert_array_equal(dataset, before)


@pytest.mark.parametrize('dtype', [np.int64, np.uint64])
def test_large_integer_observations_keep_small_differences(dtype):
    largest = int(np.iinfo(dtype).max)
    dataset = np.array([largest - 2, largest], dtype=dtype)
    assert NumericalMethods.calculate_range_value(dataset) == 2


@pytest.mark.parametrize('dataset', [
    np.array([1.5, 2.5], dtype=object),
    [2**100, 2**100 + 1],
])
def test_object_scalars_keep_their_existing_arithmetic(dataset):
    assert NumericalMethods.calculate_range_value(dataset) == 1


def test_float32_range_is_not_limited_to_float32_precision():
    dataset = np.array([-3e38, 3e38], dtype=np.float32)
    expected = float(dataset[1]) - float(dataset[0])
    with np.errstate(over='raise'):
        assert NumericalMethods.calculate_range_value(dataset) == expected


def test_range_of_a_noncontiguous_view_leaves_the_input_unchanged():
    dataset = np.array([[2.5, 8.0, -1.5], [4.0, 9.0, 1.5]])
    before = dataset.copy()
    assert NumericalMethods.calculate_range_value(dataset[:, ::2]) == 5.5
    np.testing.assert_array_equal(dataset, before)


@pytest.mark.parametrize('dataset, expected', [
    ([1.5, 2.5, 4.0], 2.5),
    ([3], 0),
    (np.array([5.0, 5.0]), 0),
])
def test_existing_one_dimensional_ranges(dataset, expected):
    assert NumericalMethods.calculate_range_value(dataset) == expected
