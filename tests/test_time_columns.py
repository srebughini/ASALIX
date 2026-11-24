from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import numpy as np
import pandas as pd

import asalix


class TimeColumnExtractionTest(TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.frame = pd.DataFrame({'reading': [101, 102, 103, 104], 'time': [1, 1, 2, 2]})

    def assert_groups(self, source, **kwargs):
        groups = asalix.extract_time_dependent_dataset(
            source, data_column_name='reading', time_column_name='time', **kwargs)
        self.assertEqual(set(groups), {1, 2})
        np.testing.assert_array_equal(groups[1], [101, 102])
        np.testing.assert_array_equal(groups[2], [103, 104])

    def test_csv_groups_measurements_by_the_time_column(self):
        path = Path(self.directory.name) / 'data.csv'
        self.frame.to_csv(path, index=False)
        self.assert_groups(str(path))

    def test_excel_groups_measurements_by_the_time_column(self):
        path = Path(self.directory.name) / 'data.xlsx'
        with pd.ExcelWriter(path, engine='openpyxl') as writer:
            pd.DataFrame({'unrelated': [0]}).to_excel(writer, sheet_name='first', index=False)
            self.frame.to_excel(writer, sheet_name='measurements', index=False)
        self.assert_groups(str(path), sheet_name='measurements')

    def test_dataframe_uses_the_same_groups_without_changing_the_input(self):
        original = self.frame.copy(deep=True)
        self.assert_groups(self.frame)
        pd.testing.assert_frame_equal(self.frame, original)
