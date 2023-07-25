from src.numerical_methods import NumericalMethods


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
    return NumericalMethods.calculate_mean_value(dataset)


def calculate_standard_deviation(dataset, population=False):
    """
    Compute the standard deviation of the dataset.
    Parameters
    ----------
    dataset: array_like
        Input data. The standard is computed over the flattened array.
    population: bool, optional
        If False the dataset is considered to be a subset of the whole data population. If True the dataset is
        considered to be the all data population.

    Returns
    -------
    sigma: dtype float
        Return the standard deviation of the dataset.
    """
    if population:
        return NumericalMethods.calculate_population_standard_deviation(dataset)

    return NumericalMethods.calculate_sample_standard_deviation(dataset)


def gaussian_fit(hist, bin, plot=False):
    """
    Gaussian fit on histogram like dataset.
    Parameters
    ----------
    hist: array
        The values of the histogram.
    bin: array of dtype float
        The bin centers (length(hist)).
    plot: bool, optional
        If False, no plot is shown. If True, dataset is plotted.

    Returns
    -------
    A: dtype float
        Return the gaussian fit coefficient.
    mu: dtype float
        Return the mean value of the dataset.
    sigma: dtype float
        Return the standard deviation of the dataset.
    """
    pass


def create_histogram(dataset, bins=10, datarange=None, density=False, gaussian_fit=False, plot=False):
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
    gaussian_fit: bool, optional
        If False, the gaussian fit is not applied. If True, the gaussian fit is applied with the following formula:
        A * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)), where A is the gaussian fit coefficient, x is the dataset, mu is
        the mean value of the dataset and sigma is the standard deviation of the dataset.
    plot: bool, optional
        If False, no plot is shown. If True, dataset is plotted.

    Returns
    -------
    hist: array
        The values of the histogram. See density for a description of the possible semantics.
    bin_edges: array of dtype float
        Return the bin edges (length(hist)+1).
    bin_centers: array of dtype float
        Return the bin centers (length(hist)).
    A: dtype float
        Return the gaussian fit coefficient, if gaussian_fit is True.
    mu: dtype float
        Return the mean value of the dataset, if gaussian_fit is True.
    sigma: dtype float
        Return the standard deviation of the dataset, if gaussian_fit is True.
    """
    pass
