import os
import sys

from asalix.dataset_extractor import DatasetExtractor
from asalix.numerical_methods import NumericalMethods
from tests.unit_test import UnitTest


def main():
    dataset_extractor = DatasetExtractor()
    dataset_result = UnitTest.check_all(os.path.join("tests", "tests.json"), dataset_extractor)

    numerical_methods = NumericalMethods()
    numerical_result = UnitTest.check_all(os.path.join("tests", "tests.json"), numerical_methods)

    return dataset_result and numerical_result


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
