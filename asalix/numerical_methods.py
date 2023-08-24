import numpy as np

from enum import Enum
from scipy.optimize import curve_fit
from scipy import stats


class NelsonRule(Enum):
    RULE1 = 1
    RULE2 = 2
    RULE3 = 3
    RULE4 = 4
    RULE5 = 5
    RULE6 = 6
    RULE7 = 7
    RULE8 = 8


class ControlChartTablesCoefficient(Enum):
    A2 = {
        2: 1.88,
        3: 1.023,
        4: 0.729,
        5: 0.577,
        6: 0.483,
        7: 0.419,
        8: 0.373,
        9: 0.337,
        10: 0.308,
        15: 0.223,
        25: 0.153}

    d2 = {
        2: 1.128,
        3: 1.693,
        4: 2.059,
        5: 2.326,
        6: 2.534,
        7: 2.704,
        8: 2.847,
        9: 2.97,
        10: 3.078,
        15: 3.472,
        25: 3.931}

    D3 = {
        7: 0.076,
        8: 0.136,
        9: 0.184,
        10: 0.223,
        15: 0.347,
        25: 0.459}

    D4 = {
        2: 3.267,
        3: 2.574,
        4: 2.282,
        5: 2.114,
        6: 2.004,
        7: 1.924,
        8: 1.864,
        9: 1.816,
        10: 1.777,
        15: 1.653,
        25: 1.541}

    A3 = {
        2: 2.659,
        3: 1.954,
        4: 1.628,
        5: 1.427,
        6: 1.287,
        7: 1.182,
        8: 1.099,
        9: 1.032,
        10: 0.975,
        15: 0.789,
        25: 0.606}

    c4 = {
        2: 0.7979,
        3: 0.8862,
        4: 0.9213,
        5: 0.94,
        6: 0.9515,
        7: 0.9594,
        8: 0.965,
        9: 0.9693,
        10: 0.9727,
        15: 0.9823,
        25: 0.9896}

    B3 = {
        6: 0.03,
        7: 0.118,
        8: 0.185,
        9: 0.239,
        10: 0.284,
        15: 0.428,
        25: 0.565}

    B4 = {
        2: 3.267,
        3: 2.568,
        4: 2.266,
        5: 2.089,
        6: 1.97,
        7: 1.882,
        8: 1.815,
        9: 1.761,
        10: 1.716,
        15: 1.572,
        25: 1.435}


class NumericalMethods:
    def __init__(self):
        """
        Class that collect all the numerical methods
        """

    @staticmethod
    def _access_to_control_chart_tables_coefficient(n, control_chart_coefficient):
        """
        Private function to easily access to the ControlChartTablesCoefficient Enum given the number of sample in the dataset
        Parameters
        ----------
        n: int
            Number of sample in the dataset
        control_chart_coefficient: Enum
            Control chart coefficient to be extracted from the Enum

        Returns
        -------
        coefficient: float
            Extracted coefficient
        """
        keys_array = np.asarray(list(control_chart_coefficient.value.keys()))
        return control_chart_coefficient.value[keys_array[np.argmin(keys_array - n)]]

    @staticmethod
    def calculate_range_value(dataset):
        """
        Compute the range of a dataset
        Parameters
        ----------
        dataset: array_like
            Input data. The mean value is computed over the flattened array.

        Returns
        -------
        range_vector: dtype float
            Return the range
        """
        return max(dataset) - min(dataset)

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
        nx = A * np.exp(-np.square(np.asarray(dataset) - mu) / (2 * np.square(sigma)))
        return nx

    @staticmethod
    def calculate_percentile(dataset, q):
        """
        Compute the q-th percentile of the dataser.
        Returns the q-th percentile(s) of the array elements.
        Parameters
        ----------
        dataset: array_like of real numbers
            Input dataset.

        q: array_like of float
            Percentage or sequence of percentages for the percentiles to compute.
            Values must be between 0 and 100 inclusive.

        Returns
        -------
        percentile: scalar or ndarray
        If q is a single percentile, then the result is a scalar. If multiple percentiles are given,
        first axis of the result corresponds to the percentiles.
        """
        return np.percentile(dataset, q)

    @staticmethod
    def calculate_z_critical_value(c):
        """
        Calculate Z-distribution critical values
        Parameters
        ----------
        c: dtype float
            Confidence Level (0-1)

        Returns
        -------
        cf: dtype float
            Confidence internal coefficient
        """
        q = 1 - (1 - c) / 2
        return stats.norm.ppf(q)

    @staticmethod
    def calculate_t_critical_value(c, n):
        """
        Calculate t-distribution critical values
        Parameters
        ----------
        c: dtype float
            Confidence Level (0-1)
        n: int
            Number of samples

        Returns
        -------
        cf: dtype float
            Confidence internal coefficient
        """
        q = 1 - (1 - c) / 2
        return stats.t.ppf(q=q, df=n)

    @staticmethod
    def calculate_population_confidence_internal(dataset, confidence_level):
        """
        Compute the confidence interval of the dataset, considered as the population
        Parameters
        ----------
        dataset: array_like
            Input data. The standard deviation is computed over the flattened array.
        confidence_level: dtype float
            Confidence Level (0-1)

        Returns
        -------
        confidence_interval: (float,float)
            Confidence internal of the dataset
        """
        n = len(dataset)
        confidence_coefficient = NumericalMethods.calculate_z_critical_value(confidence_level)
        mean_value = NumericalMethods.calculate_mean_value(dataset)
        standard_deviation = NumericalMethods.calculate_population_standard_deviation(dataset)
        variation = confidence_coefficient * standard_deviation / np.sqrt(n)
        return mean_value - variation, mean_value + variation

    @staticmethod
    def calculate_sample_confidence_internal(dataset, confidence_level):
        """
        Compute the confidence interval of the dataset, considered as a sample of the whole population
        Parameters
        ----------
        dataset: array_like
            Input data. The standard deviation is computed over the flattened array.
        confidence_level: dtype float
            Confidence Level (0-1)

        Returns
        -------
        confidence_interval: (float,float)
            Confidence internal of the dataset
        """
        n = len(dataset)
        confidence_coefficient = NumericalMethods.calculate_t_critical_value(confidence_level, n - 1)
        mean_value = NumericalMethods.calculate_mean_value(dataset)
        standard_deviation = NumericalMethods.calculate_sample_standard_deviation(dataset)
        variation = confidence_coefficient * standard_deviation / np.sqrt(n)
        return mean_value - variation, mean_value + variation

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
        return coeff[0], coeff[1], np.fabs(coeff[2])

    @staticmethod
    def anderson_darling_normality_test(dataset):
        """
        Anderson-Darling test for data coming from a particular distribution.
        The Anderson-Darling test tests the null hypothesis that a sample is drawn from a population that follows a
        particular distribution. For the Anderson-Darling test, the critical values depend on which distribution is
        being tested against. This function works for normal distributions.
        Parameters
        ----------
        dataset: array_like
            Input data.

        Returns
        -------
        p-value: float
            If the p value is < 0.05, we reject the null hypotheses that the data are from a normal distribution.
            If the p value is > 0.05, we accept the null hypotheses that the data are from a normal distribution.
        """
        res = stats.anderson(dataset)
        n = len(dataset)
        ad = res.statistic * (1 + 0.75 / n + 2.25 / (n * n))

        if ad >= 0.6:
            return np.exp(1.2937 - 5.709 * ad + 0.0186 * ad * ad)

        if ad > 0.34:
            return np.exp(0.9177 - 4.279 * ad - 1.38 * ad * ad)

        if ad > 0.2:
            return 1.0 - np.exp(-8.318 + 42.796 * ad - 59.938 * ad * ad)

        return 1.0 - np.exp(13.436 + 101.14 * ad - 223.73 * ad * ad)

    @staticmethod
    def kolmogorov_smirnov_normality_test(dataset):
        """
        Performs the (one-sample or two-sample) Kolmogorov-Smirnov test for goodness of fit for a normal distribution
        Parameters
        ----------
        dataset: array_like
            Input data.

        Returns
        -------
        p-value: float
            If the p value is < 0.05, we reject the null hypotheses that the data are from a normal distribution.
            If the p value is > 0.05, we accept the null hypotheses that the data are from a normal distribution.
        """
        mu = NumericalMethods.calculate_mean_value(dataset)
        sigma = NumericalMethods.calculate_population_standard_deviation(dataset)
        normed_dataset = (dataset - mu) / sigma
        res = stats.kstest(normed_dataset, 'norm')
        return res.pvalue

    @staticmethod
    def shapiro_wilk_normality_test(dataset):
        """
        Perform the Shapiro-Wilk test for normality.
        The Shapiro-Wilk test tests the null hypothesis that the data was drawn from a normal distribution.

        Parameters
        ----------
        dataset: array_like
            Input data.

        Returns
        -------
        p-value: float
            If the p value is < 0.05, we reject the null hypotheses that the data are from a normal distribution.
            If the p value is > 0.05, we accept the null hypotheses that the data are from a normal distribution.
        """
        res = stats.shapiro(dataset)
        return res.pvalue

    @staticmethod
    def basic_normality_test(dataset):
        """
        Test whether a sample differs from a normal distribution.
        This function tests the null hypothesis that a sample comes from a normal distribution. It is based on
        D’Agostino and Pearson’s test that combines skew and kurtosis to produce an omnibus test of normality.
        ----------
        dataset: array_like
            Input data.

        Returns
        -------
        p-value: float
            If the p value is < 0.05, we reject the null hypotheses that the data are from a normal distribution.
            If the p value is > 0.05, we accept the null hypotheses that the data are from a normal distribution.
        """
        res = stats.normaltest(dataset)
        return res.pvalue

    @staticmethod
    def perform_nelson_rule_1_test(dataset):
        """
        Perform nelson rule number 1 test for control charts
        Parameters
        ----------
        dataset: array_like
            Input data.

        Returns
        -------
        result: bool
            If True one point is more than 3 standard deviations from the mean.
        """
        sigma = NumericalMethods.calculate_sample_standard_deviation(dataset)
        mean = NumericalMethods.calculate_mean_value(dataset)
        upper_limit = mean + 3 * sigma
        lower_limit = mean - 3 * sigma
        if sum([0 if upper_limit > v > lower_limit else 1 for v in dataset]) > 0:
            return True

        return False

    @staticmethod
    def perform_nelson_rule_2_test(dataset):
        """
        Perform nelson rule number 2 test for control charts
        Parameters
        ----------
        dataset: array_like
            Input data.

        Returns
        -------
        result: bool
            if True nine (or more) points in a row are on the same side of the mean.
        """
        if len(dataset) < 9:
            return False

        mean = NumericalMethods.calculate_mean_value(dataset)
        upper_side = False
        lower_side = False
        count = 0
        for v in dataset:
            if v > mean:
                if upper_side:
                    count = count + 1
                else:
                    upper_side = True
                    lower_side = False
                    count = 1
            elif v < mean:
                if lower_side:
                    count = count + 1
                else:
                    upper_side = False
                    lower_side = True
                    count = 1
            else:
                upper_side = False
                lower_side = False
                count = 0

            if count > 8:
                return True

        return False

    @staticmethod
    def perform_nelson_rule_3_test(dataset):
        """
        Perform nelson rule number 3 test for control charts
        Parameters
        ----------
        dataset: array_like
            Input data.

        Returns
        -------
        result: bool
            if True six (or more) points in a row are continually increasing (or decreasing).
        """
        if len(dataset) < 6:
            return False

        diff_array = dataset[1:] - dataset[:-1]
        upper_side = False
        lower_side = False
        count = 0

        for v in diff_array:
            if v > 0:
                if upper_side:
                    count = count + 1
                else:
                    upper_side = True
                    lower_side = False
                    count = 1
            elif v < 0:
                if lower_side:
                    count = count + 1
                else:
                    upper_side = False
                    lower_side = True
                    count = 1
            else:
                upper_side = False
                lower_side = False
                count = 0

            if count > 5:
                return True

        return False

    @staticmethod
    def perform_nelson_rule_4_test(dataset):
        """
        Perform nelson rule number 4 test for control charts
        Parameters
        ----------
        dataset: array_like
            Input data.

        Returns
        -------
        result: bool
            if True fourteen (or more) points in a row alternate in direction, increasing then decreasing.
        """
        if len(dataset) < 14:
            return False

        diff_array = dataset[1:] - dataset[:-1]
        upper_side = False
        lower_side = False
        count = 0
        for i in range(1, len(diff_array)):
            if diff_array[i] > diff_array[i - 1]:
                if upper_side:
                    count = count + 1
                else:
                    upper_side = True
                    lower_side = False
                    count = 1
            elif diff_array[i] < diff_array[i - 1]:
                if lower_side:
                    count = count + 1
                else:
                    upper_side = False
                    lower_side = True
                    count = 1
            else:
                upper_side = False
                lower_side = False
                count = 0

            if count > 13:
                return True

        return False

    @staticmethod
    def perform_nelson_rule_5_test(dataset):
        """
        Perform nelson rule number 5 test for control charts
        Parameters
        ----------
        dataset: array_like
            Input data.

        Returns
        -------
        result: bool
            if True two (or three) out of three points in a row are more than 2 standard deviations from the mean in the same direction.
        """
        if len(dataset) < 4:
            return False

        sigma = NumericalMethods.calculate_sample_standard_deviation(dataset)
        mean = NumericalMethods.calculate_mean_value(dataset)
        upper_limit = mean + 2 * sigma
        lower_limit = mean - 2 * sigma
        upper_count = 0
        lower_count = 0

        for i in range(0, len(dataset) - 3):
            value_list = [dataset[i + j] for j in [0, 1, 2]]

            if len(list(filter(lambda x: x > upper_limit, value_list))) == 3:
                upper_count = upper_count + 1

            if len(list(filter(lambda x: x < lower_limit, value_list))) == 3:
                lower_count = lower_count + 1

            if upper_count > 1:
                return True

            if lower_count > 1:
                return True

        return False

    @staticmethod
    def perform_nelson_rule_6_test(dataset):
        """
        Perform nelson rule number 6 test for control charts
        Parameters
        ----------
        dataset: array_like
            Input data.

        Returns
        -------
        result: bool
            if True four (or five) out of five points in a row are more than 1 standard deviation from the mean in the same direction.
        """
        if len(dataset) < 6:
            return False

        sigma = NumericalMethods.calculate_sample_standard_deviation(dataset)
        mean = NumericalMethods.calculate_mean_value(dataset)
        upper_limit = mean + 1 * sigma
        lower_limit = mean - 1 * sigma
        upper_count = 0
        lower_count = 0

        for i in range(0, len(dataset) - 5):
            value_list = [dataset[i + j] for j in [0, 1, 2, 3, 4]]

            if len(list(filter(lambda x: x > upper_limit, value_list))) == 5:
                upper_count = upper_count + 1

            if len(list(filter(lambda x: x < lower_limit, value_list))) == 5:
                lower_count = lower_count + 1

            if upper_count > 3:
                return True

            if lower_count > 3:
                return True

        return False

    @staticmethod
    def perform_nelson_rule_7_test(dataset):
        """
        Perform nelson rule number 7 test for control charts
        Parameters
        ----------
        dataset: array_like
            Input data.

        Returns
        -------
        result: bool
            if True fifteen points in a row are all within 1 standard deviation of the mean on either side of the mean.
        """
        if len(dataset) < 15:
            return False

        sigma = NumericalMethods.calculate_sample_standard_deviation(dataset)
        mean = NumericalMethods.calculate_mean_value(dataset)
        upper_limit = mean + 1 * sigma
        lower_limit = mean - 1 * sigma

        for i in range(len(dataset) - 15):
            check = True
            for j in range(15):
                d = dataset[i + j]
                if d >= upper_limit or d <= lower_limit:
                    check = False
                    break

            if check:
                return True

        return False

    @staticmethod
    def perform_nelson_rule_8_test(dataset):
        """
        Perform nelson rule number 7 test for control charts
        Parameters
        ----------
        dataset: array_like
            Input data.

        Returns
        -------
        result: bool
            if True eight points in a row exist, but none within 1 standard deviation of the mean, and the points are in both directions from the mean.
        """
        if len(dataset) < 8:
            return False

        sigma = NumericalMethods.calculate_sample_standard_deviation(dataset)
        mean = NumericalMethods.calculate_mean_value(dataset)

        for i in range(len(dataset) - 8):
            check = True
            for j in range(8):
                d = dataset[i + j]
                if abs(mean - d) < sigma:
                    check = False
                    break

            if check:
                return True

        return False

    @staticmethod
    def perform_nelson_rule_test(dataset, rule_number):
        """
        Perform nelson rule test for control charts
        Parameters
        ----------
        dataset: array_like
            Input data.
        rule_number: NelsonRule Enum
            Rule number to be performed

        Returns
        -------
        result: bool
            If False dataset is ok (rule is not respected).
            If True dataset is NOT ok (rule is respected).
        """
        if rule_number == NelsonRule.RULE1:
            return NumericalMethods.perform_nelson_rule_1_test(dataset)

        if rule_number == NelsonRule.RULE2:
            return NumericalMethods.perform_nelson_rule_2_test(dataset)

        if rule_number == NelsonRule.RULE3:
            return NumericalMethods.perform_nelson_rule_3_test(dataset)

        if rule_number == NelsonRule.RULE4:
            return NumericalMethods.perform_nelson_rule_4_test(dataset)

        if rule_number == NelsonRule.RULE5:
            return NumericalMethods.perform_nelson_rule_5_test(dataset)

        if rule_number == NelsonRule.RULE6:
            return NumericalMethods.perform_nelson_rule_6_test(dataset)

        if rule_number == NelsonRule.RULE7:
            return NumericalMethods.perform_nelson_rule_7_test(dataset)

        if rule_number == NelsonRule.RULE8:
            return NumericalMethods.perform_nelson_rule_8_test(dataset)

        return None

    @staticmethod
    def calculate_control_limits(dataset, control_chart):
        """
        Calculate lower and upper control limit

        Parameters
        ----------
        dataset: dict of array of dtype float
            Input data in time dependent dataset format
        control_chart: str
            Control chart type: {'XbarR',  #Xbar-R - Range charts
                                 'XbarS',  #Xbar-S - Standard Deviation charts
                                 'mr'}     #Moving Average - Range charts

        Returns
        -------
        lcl_1: dtype float
            Lower Control Limit (LCL) for Xbar chart
        lcl_2: dtype float
            Lower Control Limit (LCL) for R or S chart
        ucl_1: dtype float
            Upper Control Limit (UCL) for Xbar chart
        ucl_2: dtype float
            Upper Control Limit (UCL) for R or S chart
        cl_1: dtype float
            Center line (CL) for Xbar chart
        cl_2: dtype float
             Center line (CL) for R or S chart
        """
        mean_vector = [NumericalMethods.calculate_mean_value(d) for d in dataset.values()]
        mean = NumericalMethods.calculate_mean_value(mean_vector)

        if control_chart == 'XbarR':
            range_vector = [NumericalMethods.calculate_range_value(d) for d in dataset.values()]
            mean_range = NumericalMethods.calculate_mean_value(range_vector[1:])
            a2 = NumericalMethods._access_to_control_chart_tables_coefficient(len(dataset),
                                                                              ControlChartTablesCoefficient.A2)
            d3 = NumericalMethods._access_to_control_chart_tables_coefficient(len(dataset),
                                                                              ControlChartTablesCoefficient.D3)

            d4 = NumericalMethods._access_to_control_chart_tables_coefficient(len(dataset),
                                                                              ControlChartTablesCoefficient.D4)

            return mean - a2 * mean_range, \
                   d3 * mean_range, \
                   mean + a2 * mean_range, \
                   d4 * mean_range, \
                   mean, \
                   mean_range

        if control_chart == 'XbarS':
            standard_deviation_vector = [NumericalMethods.calculate_sample_standard_deviation(d) for d in
                                         dataset.values()]
            mean_standard_deviation = NumericalMethods.calculate_mean_value(standard_deviation_vector)
            a3 = NumericalMethods._access_to_control_chart_tables_coefficient(len(dataset),
                                                                              ControlChartTablesCoefficient.A3)
            b3 = NumericalMethods._access_to_control_chart_tables_coefficient(len(dataset),
                                                                              ControlChartTablesCoefficient.B3)

            b4 = NumericalMethods._access_to_control_chart_tables_coefficient(len(dataset),
                                                                              ControlChartTablesCoefficient.B4)

            return mean - a3 * mean_standard_deviation, \
                   b3 * mean_standard_deviation, \
                   mean + a3 * mean_standard_deviation, \
                   b4 * mean_standard_deviation, \
                   mean, \
                   mean_standard_deviation
