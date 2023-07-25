import os

from src.dataset_extractor import DatasetExtractor
from src.numerical_methods import NumericalMethods
from tests.unit_test import UnitTest

dataset_extractor = DatasetExtractor()
UnitTest.check_all(os.path.join("tests", "tests.json"), dataset_extractor)
numerical_methods = NumericalMethods()
UnitTest.check_all(os.path.join("tests", "tests.json"), numerical_methods)
