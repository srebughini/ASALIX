import numpy as np
import pandas as pd


class DatasetExtractor:
    def __init__(self):
        """
        Class to extract the dataset from different source
        """

    @staticmethod
    def extract_dataset(data_as_generic, sheet_name=None, data_column_name=None):
        """
        Import dataset from python list,  Pandas DataFrame, .csv file, .xlsx file
        Parameters
        ----------
        data_as_generic: list,  Pandas DataFrame, str
                Input data
        sheet_name: str, optional
            Sheet name in the .xlsx file
        data_column_name: str, optional
            Column name of the dataframe that contains the data. If None the first column is returned
        Returns
        -------
        dataset: array of dtype float
            Input data in dataset format
        """
        if isinstance(data_as_generic, list):
            return DatasetExtractor.extract_dataset_from_list(data_as_generic)

        if isinstance(data_as_generic, pd.DataFrame):
            return DatasetExtractor.extract_dataset_from_dataframe(data_as_generic,
                                                                   data_column_name=data_column_name)

        if isinstance(data_as_generic, str):
            if ".csv" in data_as_generic:
                return DatasetExtractor.extract_dataset_from_csv(data_as_generic,
                                                                 data_column_name=data_column_name)

            if ".xlsx" in data_as_generic:
                return DatasetExtractor.extract_dataset_from_xlsx(data_as_generic,
                                                                  data_column_name=data_column_name,
                                                                  sheet_name=sheet_name)

        raise Exception("ASALIX: Not recognize data format")

    @staticmethod
    def extract_time_dependent_dataset(data_as_generic,
                                       data_column_name,
                                       time_column_name,
                                       sheet_name=None):
        """
        Import dataset from python list,  Pandas DataFrame, .csv file, .xlsx file
        Parameters
        ----------
        data_as_generic: Pandas DataFrame, str
                Input data
        data_column_name: str
            Column name of the dataframe that contains the data. If None the first column is returned
        time_column_name: str
            Column name of the dataframe that contains the time. If None the
        sheet_name: str, optional
            Sheet name in the .xlsx file

        Returns
        -------
        dataset: dict of array of dtype float
            Input data in time dependent dataset format
        """
        if isinstance(data_as_generic, pd.DataFrame):
            return DatasetExtractor.convert_time_and_data_vectors_to_dataset(
                DatasetExtractor.extract_dataset_from_dataframe(data_as_generic,
                                                                data_column_name=time_column_name),
                DatasetExtractor.extract_dataset_from_dataframe(data_as_generic,
                                                                data_column_name=data_column_name))

        if isinstance(data_as_generic, str):
            if ".csv" in data_as_generic:
                return DatasetExtractor.convert_time_and_data_vectors_to_dataset(
                    DatasetExtractor.extract_dataset_from_csv(data_as_generic,
                                                              data_column_name=data_column_name),
                    DatasetExtractor.extract_dataset_from_csv(data_as_generic,
                                                              data_column_name=time_column_name))

            if ".xlsx" in data_as_generic:
                return DatasetExtractor.convert_time_and_data_vectors_to_dataset(
                    DatasetExtractor.extract_dataset_from_xlsx(data_as_generic,
                                                               data_column_name=data_column_name,
                                                               sheet_name=sheet_name),
                    DatasetExtractor.extract_dataset_from_xlsx(data_as_generic,
                                                               data_column_name=time_column_name,
                                                               sheet_name=sheet_name))

        raise Exception("ASALIX: Not recognize data format")

    @staticmethod
    def extract_dataset_from_list(data_as_list):
        """
        Import dataset from python list
        Parameters
        ----------
        data_as_list: list
            Input data

        Returns
        -------
        dataset: array of dtype float
            Input data in dataset format
        """
        return np.asarray(data_as_list, dtype=np.float64)

    @staticmethod
    def extract_dataset_from_dataframe(data_as_df, data_column_name=None):
        """
        Import dataset from Pandas DataFrame
        Parameters
        ----------
        data_as_df: Pandas DataFrame
            Input data
        data_column_name: str, optional
            Column name of the dataframe that contains the data. If None the first column is returned

        Returns
        -------
        dataset: array of dtype float
            Input data in dataset format
        """
        if data_column_name is None:
            data_column_name = data_as_df.columns[0]

        return data_as_df[data_column_name].astype(float).to_numpy()

    @staticmethod
    def extract_dataset_from_csv(file_path, data_column_name=None):
        """
        Import dataset from .csv file
        Parameters
        ----------
        file_path: str
            File path
        data_column_name: str, optional
            Column name of the dataframe that contains the data. If None the first column is returned

        Returns
        -------
        dataset: array of dtype float
            Input data in dataset format
        """
        return DatasetExtractor.extract_dataset_from_dataframe(pd.read_csv(file_path),
                                                               data_column_name=data_column_name)

    @staticmethod
    def extract_dataset_from_xlsx(file_path, sheet_name=None, data_column_name=None):
        """
        Import dataset from .xlsx file
        Parameters
        ----------
        file_path: str
            File path
        sheet_name: str, optional
            Sheet name in the .xlsx file
        data_column_name: str, optional
            Column name of the dataframe that contains the data. If None the first column is returned

        Returns
        -------
        dataset: array of dtype float
            Input data in dataset format
        """
        if sheet_name is None:
            return DatasetExtractor.extract_dataset_from_dataframe(pd.read_excel(file_path),
                                                                   data_column_name=data_column_name)

        return DatasetExtractor.extract_dataset_from_dataframe(pd.read_excel(file_path, sheet_name=sheet_name),
                                                               data_column_name=data_column_name)

    @staticmethod
    def convert_time_and_data_vectors_to_dataset(time_vector, data_vector):
        """
        Convert time vector and data vector to time dependent dataset
        Parameters
        ----------
        time_vector: array_like
            Input array of time data.
        data_vector: array_like
            Input array of data.

        Returns
        -------
        dataset: dict of array of dtype float
            Input data in time dependent dataset format
        """
        df = pd.DataFrame({"time": time_vector,
                           "data": data_vector})

        return {n: d["data"].to_numpy() for n, d in df.groupby('time')}
