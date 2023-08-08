import os

from asalix.dataset_extractor import DatasetExtractor
from asalix.numerical_methods import NumericalMethods
from tests.unit_test import UnitTest

dataset_extractor = DatasetExtractor()
UnitTest.check_all(os.path.join("tests", "tests.json"), dataset_extractor)
numerical_methods = NumericalMethods()
UnitTest.check_all(os.path.join("tests", "tests.json"), numerical_methods)
