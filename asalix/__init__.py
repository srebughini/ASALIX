"""
asalix

A comprehensive collection of mathematical tools and utilities designed to support Lean Six Sigma practitioners in their process improvement journey.
"""

__version__ = "0.1.3"
__author__ = 'Stefano Rebughini'

from asalix.dataset_extractor import DatasetExtractor
from asalix.numerical_methods import NumericalMethods, NelsonRule
from asalix.plotter import Plotter
from types import SimpleNamespace

import numpy as np


def extract_dataset(data, sheet_name=None, data_column_name=None):
    """
    Import dataset from python list,  Pandas DataFrame, .csv file, .xlsx file
    Parameters
    ----------
    data: list,  Pandas DataFrame, str
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
    return DatasetExtractor.extract_dataset(data, sheet_name=sheet_name, data_column_name=data_column_name)


def extract_time_dependent_dataset(data,
                                   data_column_name,
                                   time_column_name,
                                   sheet_name=None):
    """
    Import dataset from python list,  Pandas DataFrame, .csv file, .xlsx file
    Parameters
    ----------
    data: Pandas DataFrame, str
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
    return DatasetExtractor.extract_time_dependent_dataset(data, data_column_name, time_column_name,
                                                           sheet_name=sheet_name)


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


def calculate_confidence_interval(dataset, confidence_level, population=False):
    """
    Compute the confidence internval of the dataset.
    Parameters
    ----------
    dataset: array_like
        Input data. The standard is computed over the flattened array.
    confidence_level: dtype float
        Confidence Level (0-1)
    population: bool, optional
        If False the dataset is considered to be a subset of the whole data population. If True the dataset is
        considered to be the all data population.

    Returns
    -------
    confidence_interval: (float,float)
        Confidence internal of the dataset
    """
    if population:
        return NumericalMethods.calculate_population_confidence_internal(dataset, confidence_level)

    return NumericalMethods.calculate_sample_confidence_internal(dataset, confidence_level)


def normality_test(dataset, test='basic'):
    """
    Normality test on the dataset
    Parameters
    ----------
    dataset: array_like
        Input data.
    test: str, optional
        Normality test type: {'basic', 'anderson-darling', 'kolmogorov_smirnov', 'shapiro_wilk'}

    Returns
    -------
    p-value: float
        If the p value is < 0.05, we reject the null hypotheses that the data are from a normal distribution.
        If the p value is > 0.05, we accept the null hypotheses that the data are from a normal distribution.
    """
    if test == 'anderson-darling':
        return NumericalMethods.anderson_darling_normality_test(dataset)

    if test == 'kolmogorov_smirnov':
        return NumericalMethods.kolmogorov_smirnov_normality_test(dataset)

    if test == 'shapiro_wilk':
        return NumericalMethods.shapiro_wilk_normality_test(dataset)

    return NumericalMethods.basic_normality_test(dataset)


def normal_distribution_fit(dataset,
                            bins=10,
                            datarange=None,
                            density=False,
                            normal_distribution_test='basic'):
    """
    Gaussian fit on dataset.
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
    normal_distribution_test: str, optional
        Normality test type: {'basic', 'anderson-darling', 'kolmogorov_smirnov', 'shapiro_wilk'}
    Returns
    -------
    p-value: dtype float
        If the p value is < 0.05, we reject the null hypotheses that the data are from a normal distribution.
        If the p value is > 0.05, we accept the null hypotheses that the data are from a normal distribution.
    a: dtype float
        Return the gaussian fit coefficient.
    mu: dtype float
        Return the mean value of the dataset.
    sigma: dtype float
        Return the standard deviation of the dataset.
    """
    p_value = normality_test(dataset, test=normal_distribution_test)
    mu0 = NumericalMethods.calculate_mean_value(dataset)
    sigma0 = NumericalMethods.calculate_sample_standard_deviation(dataset)

    hist, _, bin_centres = NumericalMethods.create_histogram(dataset,
                                                             bins=bins,
                                                             datarange=datarange,
                                                             density=density)
    a, mu, sigma = NumericalMethods.normal_distribution_fit(hist, bin_centres, 1, mu0, sigma0)
    return SimpleNamespace(p_value=p_value,
                           normal_coefficient=a,
                           mean_value=mu,
                           standard_deviation=sigma)


def create_histogram(dataset,
                     bins=10,
                     datarange=None,
                     density=False,
                     normal_distribution_fitting=False,
                     normal_distribution_test='basic',
                     plot=False,
                     fig_number=1):
    """
    Create the histogram of a dataset.
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
    normal_distribution_fitting: bool, optional
        If False, the gaussian fit is not applied. If True, the gaussian fit is applied with the following formula:
        A * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)), where A is the gaussian fit coefficient, x is the dataset, mu is
        the mean value of the dataset and sigma is the standard deviation of the dataset.
    normal_distribution_test: str, optional
        Normality test type: {'basic', 'anderson-darling', 'kolmogorov_smirnov', 'shapiro_wilk'}
    plot: bool, optional
        If False, no plot is shown. If True, dataset is plotted.
    fig_number: int, optional
        Figure number

    Returns
    -------
    hist: array
        The values of the histogram. See density for a description of the possible semantics.
    bin_edges: array of dtype float
        Return the bin edges (length(hist)+1).
    bin_centers: array of dtype float
        Return the bin centers (length(hist)).
    a: dtype float
        Return the gaussian fit coefficient, if gaussian_fit is True.
    mu: dtype float
        Return the mean value of the dataset, if gaussian_fit is True.
    sigma: dtype float
        Return the standard deviation of the dataset, if gaussian_fit is True.
    """
    hist, bin_edges, bin_centres = NumericalMethods.create_histogram(dataset,
                                                                     bins=bins,
                                                                     datarange=datarange,
                                                                     density=density)

    p_value = normality_test(dataset, test=normal_distribution_test)

    fig, ax = Plotter.create_figure(fig_number)
    fig, ax = Plotter.add_title(fig, ax, "Histogram")
    fig, ax = Plotter.add_histogram(fig, ax, dataset, bins=bins, datarange=datarange, density=density)

    if density:
        fig, ax = Plotter.add_axes_labels(fig, ax, "Values", "Density")
    else:
        fig, ax = Plotter.add_axes_labels(fig, ax, "Values", "Occurrence")

    fig, ax = Plotter.add_normal_distribution_text_box(fig,
                                                       ax,
                                                       NumericalMethods.calculate_mean_value(dataset),
                                                       NumericalMethods.calculate_population_standard_deviation(
                                                           dataset),
                                                       len(dataset),
                                                       p_value)

    if not normal_distribution_fitting:
        if plot:
            Plotter.show(fig, ax)
        return SimpleNamespace(hist=hist, bin_edges=bin_edges, bin_centres=bin_centres)

    a, mu, sigma = NumericalMethods.normal_distribution_fit(hist,
                                                            bin_centres,
                                                            1,
                                                            NumericalMethods.calculate_mean_value(dataset),
                                                            NumericalMethods.calculate_population_standard_deviation(
                                                                dataset))

    fig, ax = Plotter.add_normal_distribution_line(fig, ax, a, mu, sigma)

    if plot:
        Plotter.show(fig, ax)
    return SimpleNamespace(hist=hist,
                           bin_edges=bin_edges,
                           bin_centres=bin_centres,
                           normal_coefficient=a,
                           mean_value=mu,
                           standard_deviation=sigma)


def create_quartiles(dataset, plot=False, fig_number=1):
    """
    Create the quartiles of the dataset
    Parameters
    ----------
    dataset: array_like
        Input data. The quartiles are computed over the flattened array.
    plot: bool, optional
        If False, no plot is shown. If True, dataset is plotted.
    fig_number: int, optional
        Figure number
    Returns
    -------
    minimum: dtype float
        Minimum value of dataset. 0%
    first: dtype float
        1st quartile - 25%
    median, second: dtype float
        2nd quartile - 50%
    third: dtype float
        3rd quartile - 75%
    maximum, fourth: dtype float
        4th quartile - 100%
    """
    minimum = np.min(dataset)
    first_quartile = NumericalMethods.calculate_percentile(dataset, 25)
    median = NumericalMethods.calculate_percentile(dataset, 50)
    third_quartile = NumericalMethods.calculate_percentile(dataset, 75)
    maximum = np.max(dataset)

    if plot:
        fig, ax = Plotter.create_figure(fig_number)
        fig, ax = Plotter.add_title(fig, ax, "Boxplot")
        fig, ax = Plotter.add_boxplot(fig, ax, dataset)
        fig, ax = Plotter.add_axes_labels(fig, ax, "", "Values")
        fig, ax = Plotter.add_quartile_text_box(fig, ax, minimum, first_quartile, median, third_quartile, maximum)
        Plotter.show(fig, ax)

    return SimpleNamespace(minimum=minimum,
                           first=first_quartile,
                           median=median,
                           second=median,
                           third=third_quartile,
                           maximum=maximum,
                           fourth=maximum)


def check_dataset_using_nelson_rules(dataset):
    """
    Check the dataset using the Nelson Rules (https://en.wikipedia.org/wiki/Nelson_rules)
    Parameters
    ----------
    dataset: array_like
        Input data.

    Returns
    -------
    rule1: bool
        if True one point is more than 3 standard deviations from the mean.
    rule2: bool
        if True nine (or more) points in a row are on the same side of the mean.
    rule3: bool
        if True six (or more) points in a row are continually increasing (or decreasing).
    rule4: bool
        if True fourteen (or more) points in a row alternate in direction, increasing then decreasing.
    rule5: bool
        if True two (or three) out of three points in a row are more than 2 standard deviations from the mean in the same direction.
    rule6: bool
        if True four (or five) out of five points in a row are more than 1 standard deviation from the mean in the same direction.
    rule7: bool
        if True fifteen points in a row are all within 1 standard deviation of the mean on either side of the mean.
    rule8: bool
        if True eight points in a row exist, but none within 1 standard deviation of the mean, and the points are in both directions from the mean.
    """
    return SimpleNamespace(rule1=NumericalMethods.perform_nelson_rule_test(dataset, NelsonRule.RULE1),
                           rule2=NumericalMethods.perform_nelson_rule_test(dataset, NelsonRule.RULE2),
                           rule3=NumericalMethods.perform_nelson_rule_test(dataset, NelsonRule.RULE3),
                           rule4=NumericalMethods.perform_nelson_rule_test(dataset, NelsonRule.RULE4),
                           rule5=NumericalMethods.perform_nelson_rule_test(dataset, NelsonRule.RULE5),
                           rule6=NumericalMethods.perform_nelson_rule_test(dataset, NelsonRule.RULE6),
                           rule7=NumericalMethods.perform_nelson_rule_test(dataset, NelsonRule.RULE7),
                           rule8=NumericalMethods.perform_nelson_rule_test(dataset, NelsonRule.RULE7))


def create_control_charts(dataset, control_chart, plot=False, fig_number=1):
    """
    Create control chart on the dataset
    Parameters
    ----------
    dataset: dict of array of dtype float
        Input data in time dependent dataset format
    control_chart: str
        Control chart type: {'XbarR',  #Xbar-R - Range charts
                             'XbarS',  #Xbar-S - Standard Deviation charts}

    plot: bool, optional
        If False, no plot is shown. If True, dataset is plotted.
    fig_number: int, optional
        Figure number
    Returns
    -------
    Xbar lcl, Xbar ucl: dtype float
        Lower Control Limit
    R/S lcl, R/S ucl: dtype float
        Upper Control Limit
    """
    lcl_xbar, lcl_range, ucl_xbar, ucl_range, cl_xbar, cl_range = NumericalMethods.calculate_control_limits(dataset,
                                                                                                            control_chart)

    if plot:
        fig_xbar, ax_xbar = Plotter.create_figure(fig_number)
        fig_xbar, ax_xbar = Plotter.add_title(fig_xbar, ax_xbar, "Xbar Chart")
        fig_xbar, ax_xbar = Plotter.add_control_limit_line(fig_xbar, ax_xbar, dataset, lcl_xbar, fmt='r')
        fig_xbar, ax_xbar = Plotter.add_control_limit_line(fig_xbar, ax_xbar, dataset, ucl_xbar, fmt='r')
        fig_xbar, ax_xbar = Plotter.add_control_limit_line(fig_xbar, ax_xbar, dataset, cl_xbar, fmt='--k')
        x = list(dataset.keys())
        y_bar = [NumericalMethods.calculate_mean_value(dataset[k]) for k in x]
        fig_xbar, ax_xbar = Plotter.add_line(fig_xbar, ax_xbar, x, y_bar, fmt='-ob')

        fig_range, ax_range = Plotter.create_figure(fig_number + 1)
        if control_chart == "XbarR":
            fig_range, ax_range = Plotter.add_title(fig_range, ax_range, "Range Chart")
            y_range = [NumericalMethods.calculate_range_value(dataset[k]) for k in x]

        if control_chart == "XbarS":
            fig_range, ax_range = Plotter.add_title(fig_range, ax_range, "Standard Deviation Chart")
            y_range = [NumericalMethods.calculate_sample_standard_deviation(dataset[k]) for k in x]

        fig_range, ax_range = Plotter.add_control_limit_line(fig_range, ax_range, dataset, lcl_range, fmt='r')
        fig_range, ax_range = Plotter.add_control_limit_line(fig_range, ax_range, dataset, ucl_range, fmt='r')
        fig_range, ax_range = Plotter.add_control_limit_line(fig_range, ax_range, dataset, cl_range, fmt='--k')
        fig_range, ax_range = Plotter.add_line(fig_range, ax_range, x, y_range, fmt='-ob')

        Plotter.show(fig_xbar, ax_xbar)
        Plotter.show(fig_range, ax_range)

    return SimpleNamespace(lcl=lcl_xbar, ucl=ucl_xbar, cl=cl_xbar), \
           SimpleNamespace(lcl=lcl_range, ucl=ucl_range, cl=cl_range)
