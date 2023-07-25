import numpy as np


class NumericalMethods:
    def __init__(self):
        """
        Class that collect all the numerical methods
        """

    @staticmethod
    def calculate_mean_value(dataset):
        """
        Compute the arithmetic mean of the dataset.
        Parameters
        ----------
        dataset: array_like
            Input data. The mean value is computed over the flattened array.

        Returns
        -------
        mu: dtype float
            Return the mean value of the dataset.
        """
        return np.mean(dataset)

    @staticmethod
    def calculate_population_standard_deviation(dataset):
        """
        Compute the standard deviation of the dataset, considered as the population
        Parameters
        ----------
        dataset: array_like
            Input data. The standard deviation is computed over the flattened array.

        Returns
        -------
        mu: dtype float
            Return the standard deviation of the dataset.
        """
        return np.std(dataset, ddof=0)

    @staticmethod
    def calculate_sample_standard_deviation(dataset):
        """
        Compute the standard deviation of the dataset, considered as a sample of the whole population
        Parameters
        ----------
        dataset: array_like
            Input data. The standard deviation is computed over the flattened array.

        Returns
        -------
        mu: dtype float
            Return the standard deviation of the dataset.
        """
        return np.std(dataset, ddof=1)

    @staticmethod
    def gauss(x, A, mu, sigma):
        """
        Gaussian function
        :param x: Independent variable
        :param A: Gaussian fit coefficient
        :param mu: Mean value
        :param sigma: Standard deviation
        :return:
        """
        return A * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

    @staticmethod
    def get_histogram_values(x, density=False, range=None):
        """
        Get histogram values
        :param x: Data
        :param density: If False, the result will contain the number of samples in each bin. If True, the result is the value of the probability density function at the bin, normalized such that the integral over the range is 1. Note that the sum of the histogram values will not be equal to 1 unless bins of unity width are chosen; it is not a probability mass function
        :param range:The lower and upper range of the bins. If not provided, range is simply (a.min(), a.max()). Values outside the range are ignored. The first element of the range must be less than or equal to the second. range affects the automatic bin computation as well. While bin width is computed to be optimal based on the actual data within range, the bin count will fill the entire range including portions containing no data.
        :return: Center of bins, Occurance
        """
        hist, bin_edges = np.histogram(x, density=density, range=range)
        bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2
        return bin_centres, hist

    @staticmethod
    def gaussian_fit(x, A0=None, mu0=None, sigma0=None, range=None, density=False):
        """
        Gaussian fit of data
        :param x: Data
        :param A0: Guessed A value
        :param mu0: Guessed mu value
        :param sigma0: Guessed sigma value
        :param range: Range to generate the histogram
        :param density: Density to generate the historgram
        :return: A, mu, sigma
        """
        bin, hist = NumericalMethods.get_histogram_values(x, range=range, density=density)

        if A0 is None:
            A0 = 1.

        if sigma0 is None:
            sigma0 = np.std(x)

        if mu0 is None:
            mu0 = np.mean(x)

        coeff, _ = curve_fit(NumericalMethods.gauss, bin, hist, p0=[A0, mu0, sigma0])

        return coeff[0], coeff[1], coeff[2]
