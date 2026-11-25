from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import numpy as np
import pandas as pd

from asalix.dataset_extractor import DatasetExtractor


class FileFormatDetectionTest(TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        # Equal columns isolate format routing from the separately fixed
        # time/measurement argument order in file imports.
        self.frame = pd.DataFrame({'value': [1, 2, 2], 'time': [1, 2, 2]})

    def assert_imports(self, path):
        np.testing.assert_array_equal(
            DatasetExtractor.extract_dataset(str(path), data_column_name='value'),
            [1, 2, 2])
        groups = DatasetExtractor.extract_time_dependent_dataset(str(path), 'value', 'time')
        self.assertEqual(set(groups), {1, 2})
        np.testing.assert_array_equal(groups[1], [1])
        np.testing.assert_array_equal(groups[2], [2, 2])

    def test_uppercase_csv_extension(self):
        path = self.root / 'data.CSV'
        self.frame.to_csv(path, index=False)
        self.assert_imports(path)

    def test_uppercase_excel_extension(self):
        path = self.root / 'data.XLSX'
        # The explicit engine allows pandas to write an uppercase suffix.
        self.frame.to_excel(path, engine='openpyxl', index=False)
        self.assert_imports(path)

    def test_csv_in_a_directory_name_does_not_select_the_csv_reader(self):
        directory = self.root / 'archive.csv'
        directory.mkdir()
        path = directory / 'data.xlsx'
        self.frame.to_excel(path, engine='openpyxl', index=False)
        self.assert_imports(path)

    def test_ordinary_csv_remains_supported(self):
        path = self.root / 'data.csv'
        self.frame.to_csv(path, index=False)
        self.assert_imports(path)
