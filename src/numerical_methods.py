import numpy as np
from scipy.optimize import curve_fit


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
    def calculate_normal_distribution(dataset, A, mu, sigma):
        """
        Function representing the normal distribution
        Parameters
        ----------
        dataset: array_like
            Input data.
        A: dtype float
            Gaussian fit coefficient.
        mu: dtype float
            Mean value of the dataset.
        sigma: dtype float
            Standard deviation of the dataset.

        Returns
        -------
        nx: array_like
            Normal distribution of the dataset
        """
        nx = A * np.exp(-(np.asarray(dataset) - mu) ** 2 / (2 * sigma ** 2))
        return nx

    @staticmethod
    def create_histogram(dataset, bins=10, datarange=None, density=False):
        """
        Compute the histogram of a dataset.
        Parameters
        ----------
        dataset: array_like
            Input data. The histogram is computed over the flattened array.
        bins: int or sequence of scalars or str, optional
            If bins is an int, it defines the number of equal-width bins in the given range (10, by default).
            If bins is a sequence, it defines a monotonically increasing array of bin edges, including the rightmost edge,
            allowing for non-uniform bin widths.
        datarange: (float, float), optional
            The lower and upper range of the bins. If not provided, range is simply (a.min(), a.max()). Values outside the
            range are ignored. The first element of the range must be less than or equal to the second. range affects the
            automatic bin computation as well. While bin width is computed to be optimal based on the actual data within
            range, the bin count will fill the entire range including portions containing no data.
        density: bool, optional
            If False, the result will contain the number of samples in each bin. If True, the result is the value of the
            probability density function at the bin, normalized such that the integral over the range is 1. Note that the
            sum of the histogram values will not be equal to 1 unless bins of unity width are chosen;
            it is not a probability mass function.

        Returns
        -------
        hist: array
            The values of the histogram. See density for a description of the possible semantics.
        bin_edges: array of dtype float
            Return the bin edges (length(hist)+1).
        bin_centers: array of dtype float
            Return the bin centers (length(hist)).
        """
        hist, bin_edges = np.histogram(dataset, bins=bins, density=density, range=datarange)
        bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2
        return hist, bin_edges, bin_centres

    @staticmethod
    def normal_distribution_fit(hist, bin, A0, mu0, sigma0):
        """
        Fit dataset with a normal distribution
        Parameters
        ----------
        hist: array
            The values of the histogram.
        bin: array of dtype float
            The bin centers (length(hist)).
        A0: dtype float
            Gaussian fit coefficient first guess.
        mu0: dtype float
            Mean value of the dataset first guess.
        sigma0: dtype float
            Standard deviation of the dataset first guess.

        Returns
        -------
        A: dtype float
            Return the gaussian fit coefficient.
        mu: dtype float
            Return the mean value of the dataset.
        sigma: dtype float
            Return the standard deviation of the dataset.
        """
        coeff, _ = curve_fit(NumericalMethods.calculate_normal_distribution, bin, hist, p0=[A0, mu0, sigma0])

        return coeff[0], coeff[1], coeff[2]
