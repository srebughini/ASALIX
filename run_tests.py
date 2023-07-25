import os

from src.dataset_extractor import DatasetExtractor
from tests.unit_test import UnitTest

dataset_extractor = DatasetExtractor()
UnitTest.check_all(os.path.join("tests", "tests.json"), dataset_extractor)
