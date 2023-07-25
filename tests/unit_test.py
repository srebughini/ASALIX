from enum import Enum
from termcolor import colored

import json
import os
import numpy as np
import pandas as pd


class ArgsFormat(Enum):
    """
    Enum describing the ArgsFormat
    """
    NONE = "none"
    TUPLE = "tuple"
    FLOAT = "float"
    ARRAY = "array"
    MATRIX = "matrix"
    ENUM = "enum"
    DATAFRAME = "dataframe"
    CSV = "csv"
    XLSX = "xlsx"


class UnitTestScreenPrinter:
    @staticmethod
    def comparison_on_screen(outputs, results, results_format):
        """
        Function to print the comparison on screen
        Parameters
        ----------
        outputs: Anything
            Output of the function
        results: Anything
            Expected output of the function
        results_format: Enum
            Args format

        Returns
        -------
        None
        """
        msg = 'Output value: {}\nExpected value: {}'
        if results_format == ArgsFormat.ENUM:
            print(msg.format(int(outputs), int(results)))
        else:
            print(msg.format(outputs, results))

    @staticmethod
    def generic_print_on_screen(cls, f, main_msg, color):
        """
        Generic function to print on screen the results of a unit test
        Parameters
        ----------
        cls: Python object
            Class tested
        f: Python object
            Function tested
        main_msg: str
            Main message to print on screen
        color: str
            Color of the message

        Returns
        -------
        None
        """
        if not isinstance(f, str):
            function_name = f.__name__
        else:
            function_name = f

        tested_function = '{}::{}'.format(cls.__class__.__name__, function_name)
        msg = 'ASALIX::{}{} --> {}'.format(tested_function, ' ' * (60 - len(tested_function)), main_msg)
        print(colored(msg, color))

    @staticmethod
    def not_ok_test_on_screen(cls, f):
        """
        Function to print NOT OK test on screen
        Parameters
        ----------
        cls: Python object
            Class tested
        f: Python object
            Function tested

        Returns
        -------
        None
        """
        UnitTestScreenPrinter.generic_print_on_screen(cls, f, "NOT OK", "red")

    @staticmethod
    def ok_test_on_screen(cls, f):
        """
        Function to print error on screen
        Parameters
        ----------
        cls: Python object
            Class tested
        f: Python object
            Function tested

        Returns
        -------
        None
        """
        UnitTestScreenPrinter.generic_print_on_screen(cls, f, "OK", "green")


class UnitTest:
    def __init__(self):
        """
        Unit test class
        """

    @staticmethod
    def run_generic_function(f, args, args_format):
        """
        Function to run a generic function
        Parameters
        ----------
        f: Python object
            Function to be run
        args: array_like
            Args of the function
        args_format: Enum
            Args format

        Returns
        -------
        Returns output of the function
        """
        if args_format == ArgsFormat.NONE:
            return f()

        if args_format == ArgsFormat.TUPLE:
            return f(*args)

        if args_format == ArgsFormat.DATAFRAME:
            df = pd.DataFrame(args)
            return f(df)

        return f(args)

    @staticmethod
    def check_float(f, results, args, args_format):
        """
        Function to check and compare floats
        Parameters
        ----------
        f: Python object
            Function to be run
        results: Anything
            Expected output of the function
        args: array_like
            Args of the function
        args_format: str
            Args type

        Returns
        -------
        check: Bool
            If True the test is passed, if False the test is failed
        outputs: Anything
            Output of the function
        results: Anything
            Expected output of the function

        """
        outputs = UnitTest.run_generic_function(f, args, ArgsFormat(args_format))
        return np.fabs(outputs - results) < 1.e-12, outputs, results

    @staticmethod
    def check_array(f, results, args, args_format, atol=1.e-02, rtol=1.e-02):
        """
        Function to check and compare arrays
        Parameters
        ----------
        f: Python object
            Function to be run
        results: Anything
            Expected output of the function
        args: array_like
            Args of the function
        args_format: str
            Args type
        atol: float, optional
            Absolute tolerance for the comparison
        rtol: float, optional
            Relative tolerance for the comparison

        Returns
        -------
        check: Bool
            If True the test is passed, if False the test is failed
        outputs: Anything
            Output of the function
        results: Anything
            Expected output of the function

        """
        outputs = UnitTest.run_generic_function(f, args, ArgsFormat(args_format))

        outputs_as_array = np.asarray(outputs)
        results_as_array = np.asarray(results)

        if outputs_as_array.shape == results_as_array.shape:
            return np.allclose(outputs, results, atol=atol, rtol=rtol), outputs, results

        return False, outputs, results

    @staticmethod
    def check_matrix(f, results, args, args_format, atol=1.e-02, rtol=1.e-02):
        """
        Function to check and compare matrix
        Parameters
        ----------
        f: Python object
            Function to be run
        results: Anything
            Expected output of the function
        args: array_like
            Args of the function
        args_format: str
            Args type
        atol: float, optional
            Absolute tolerance for the comparison
        rtol: float, optional
            Relative tolerance for the comparison

        Returns
        -------
        check: Bool
            If True the test is passed, if False the test is failed
        outputs: Anything
            Output of the function
        results: Anything
            Expected output of the function

        """
        outputs = UnitTest.run_generic_function(f, args, ArgsFormat(args_format))

        if len(outputs) == len(results):
            for i, r in enumerate(results):
                if not np.allclose(outputs[i], r, atol=atol, rtol=rtol):
                    return False, outputs, results

            return True, outputs, results

        return False, outputs, results

    @staticmethod
    def check_enum(f, results, args, args_format):
        """
        Function to check and compare enum
        Parameters
        ----------
        f: Python object
            Function to be run
        results: Anything
            Expected output of the function
        args: array_like
            Args of the function
        args_format: str
            Args type

        Returns
        -------
        check: Bool
            If True the test is passed, if False the test is failed
        outputs: Anything
            Output of the function
        results: Anything
            Expected output of the function

        """
        outputs = UnitTest.run_generic_function(f, args, ArgsFormat(args_format))
        return int(outputs) == results, outputs, results

    @staticmethod
    def check_others(f, results, args, args_format):
        """
        Function to check and compare not float, array or enum
        Parameters
        ----------
        f: Python object
            Function to be run
        results: Anything
            Expected output of the function
        args: array_like
            Args of the function
        args_format: str
            Args type

        Returns
        -------
        check: Bool
            If True the test is passed, if False the test is failed
        outputs: Anything
            Output of the function
        results: Anything
            Expected output of the function

        """
        outputs = UnitTest.run_generic_function(f, args, ArgsFormat(args_format))
        return outputs == results, outputs, results

    @staticmethod
    def check_function(cls, f, testing_parameters, print_on_screen):
        """
        Function to test a single function
        Parameters
        ----------
        cls: Python object
            Class to be tested
        f: Python object
            Function to be tested
        testing_parameters: dict
            Dictionary of the input/output values and args format for the function to be tested
        print_on_screen: Bool
            If True the output of the test is printed on screen. If False the test is performed in silence mode.

        Returns
        -------
        check: Bool
            If True the test is passed, if False the test is failed
        """
        args = testing_parameters["input"]["value"]
        results = testing_parameters["output"]["value"]
        args_format = testing_parameters["input"]["format"]
        results_format = testing_parameters["output"]["format"]

        if results_format == ArgsFormat.FLOAT.value:
            check, outputs, results = UnitTest.check_float(f, results, args, args_format)
        elif results_format == ArgsFormat.ARRAY.value:
            check, outputs, results = UnitTest.check_array(f, results, args, args_format)
        elif results_format == ArgsFormat.ENUM.value:
            check, outputs, results = UnitTest.check_enum(f, results, args, args_format)
        elif results_format == ArgsFormat.MATRIX.value:
            check, outputs, results = UnitTest.check_matrix(f, results, args, args_format)
        else:
            check, outputs, results = UnitTest.check_others(f, results, args, args_format)

        if print_on_screen:
            if check:
                UnitTestScreenPrinter.ok_test_on_screen(cls, f)
            else:
                UnitTestScreenPrinter.not_ok_test_on_screen(cls, f)
                UnitTestScreenPrinter.comparison_on_screen(outputs, results, results_format)

        return check

    @staticmethod
    def check_all(tests_file_path, cls, print_on_screen=True):
        """
        Check all functions present in the JSON file
        Parameters
        ----------
        tests_file_path: str
            File path with the tests to be performed in JSON format
        cls: Python object
            Class to be tested
        print_on_screen: Bool, optional
            If True the output of the test is printed on screen. If False the test is performed in silence mode.

        Returns
        -------
        None
        """
        with open(tests_file_path) as json_file:
            solution_dict = json.load(json_file)

        for function, testing_parameters in solution_dict[cls.__class__.__name__].items():
            f = getattr(cls, function)
            UnitTest.check_function(cls, f, testing_parameters, print_on_screen)
