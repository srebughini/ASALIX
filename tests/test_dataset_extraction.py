from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from asalix import extract_dataset


def test_list_extraction_preserves_observations():
    np.testing.assert_array_equal(extract_dataset([1, 2, 3, 4]), [1, 2, 3, 4])


def test_dataframe_extraction_defaults_to_the_first_column():
    frame = pd.DataFrame({'a': [1, 2, 3, 4], 'b': [3, 4, 5, 7]})

    np.testing.assert_array_equal(extract_dataset(frame), [1, 2, 3, 4])


@pytest.mark.parametrize('filename', ['test1.csv', 'test2.xlsx'])
def test_file_extraction_defaults_to_the_first_column(filename):
    path = Path(__file__).parent / 'files' / filename

    np.testing.assert_array_equal(extract_dataset(str(path)), [1, 3, 5, 7])
