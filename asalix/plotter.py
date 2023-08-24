import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnchoredText

from asalix.numerical_methods import NumericalMethods


class Plotter:
    def __init__(self):
        """
        Class to plot
        """

    @staticmethod
    def create_figure(figure_number):
        """
        Create the figure and axis object of matplotlib
        Parameters
        ----------
        figure_number: int
            Figure number associated with the plot

        Returns
        -------
        fig: Figure
            Matplotlib figure class
        ax: Axes or array of Axes
            ax can be either a single Axes object, or an array of Axes objects if more than one subplot was created.
            In this case only ONE subplot is created
        """
        fig = plt.figure(figure_number)
        ax = fig.add_subplot(111)
        return fig, ax

    @staticmethod
    def add_histogram(fig, ax, dataset, bins, density, datarange):
        """
        Add histogram plot
        Parameters
        ----------
        fig: Figure
            Matplotlib figure class
        ax: Axes or array of Axes
            ax can be either a single Axes object, or an array of Axes objects if more than one subplot was created.
            In this case only ONE subplot is created
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
        fig: Figure
            Matplotlib figure class
        ax: Axes or array of Axes
            ax can be either a single Axes object, or an array of Axes objects if more than one subplot was created.
            In this case only ONE subplot is created
        """
        ax.hist(dataset, bins=bins, range=datarange, density=density)
        return fig, ax

    @staticmethod
    def add_line(fig, ax, x, y, fmt=None, label=None):
        """
        Add line plot
        Parameters
        ----------
        fig: Figure
            Matplotlib figure class
        ax: Axes or array of Axes
            ax can be either a single Axes object, or an array of Axes objects if more than one subplot was created.
            In this case only ONE subplot is created
        x, y: array-like or scalar
            The horizontal / vertical coordinates of the data points. x values are optional and default to
            range(len(y)). Commonly, these parameters are 1D arrays.They can also be scalars, or two-dimensional
            (in that case, the columns represent separate data sets).These arguments cannot be passed as keywords.
        fmt: str, optional
            A format string, e.g. 'ro' for red circles. See the matplotlib section for a full description of the format
            strings. Format strings are just an abbreviation for quickly setting basic line properties. All of these and
            more can also be controlled by keyword arguments. This argument cannot be passed as keyword.
        label: str, optional
            Label associate with the data points

        Returns
        -------
        fig: Figure
            Matplotlib figure class
        ax: Axes or array of Axes
            ax can be either a single Axes object, or an array of Axes objects if more than one subplot was created.
            In this case only ONE subplot is created
        """
        if fmt is None:
            ax.plot(x, y, label=label)
        else:
            ax.plot(x, y, fmt, label=label)

        return fig, ax

    @staticmethod
    def add_boxplot(fig, ax, dataset, labels=[""]):
        """
        Add boxplot
        Parameters
        ----------
        fig: Figure
            Matplotlib figure class
        ax: Axes or array of Axes
            ax can be either a single Axes object, or an array of Axes objects if more than one subplot was created.
            In this case only ONE subplot is created
        dataset: array_like
            Input data. The histogram is computed over the flattened array.
        labels : sequence, optional
            Labels for each dataset. Length must be compatible with dimensions of x.

        Returns
        -------
        fig: Figure
            Matplotlib figure class
        ax: Axes or array of Axes
            ax can be either a single Axes object, or an array of Axes objects if more than one subplot was created.
            In this case only ONE subplot is created
        """
        ax.boxplot(dataset, labels=labels)
        return fig, ax

    @staticmethod
    def add_axes_labels(fig, ax, x_label, y_label):
        """
        Add axes labels
        Parameters
        ----------
        fig: Figure
            Matplotlib figure class
        ax: Axes or array of Axes
            ax can be either a single Axes object, or an array of Axes objects if more than one subplot was created.
            In this case only ONE subplot is created
        x_label, y_label: str
            The horizontal / vertical axes labels.

        Returns
        -------
        fig: Figure
            Matplotlib figure class
        ax: Axes or array of Axes
            ax can be either a single Axes object, or an array of Axes objects if more than one subplot was created.
            In this case only ONE subplot is created
        """
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        return fig, ax

    @staticmethod
    def add_title(fig, ax, title):
        """
        Add title
        Parameters
        ----------
        fig: Figure
            Matplotlib figure class
        ax: Axes or array of Axes
            ax can be either a single Axes object, or an array of Axes objects if more than one subplot was created.
            In this case only ONE subplot is created
        title: str
            Plot title.

        Returns
        -------
        fig: Figure
            Matplotlib figure class
        ax: Axes or array of Axes
            ax can be either a single Axes object, or an array of Axes objects if more than one subplot was created.
            In this case only ONE subplot is created
        """
        ax.set_title(title)
        return fig, ax

    @staticmethod
    def add_text_box(fig, ax, text, loc=2):
        """
        Add text box
        Parameters
        ----------
        fig: Figure
            Matplotlib figure class
        ax: Axes or array of Axes
            ax can be either a single Axes object, or an array of Axes objects if more than one subplot was created.
            In this case only ONE subplot is created
        text: str
            Text to be added.
        loc: int, optional
            Location of the text.
            {'upper right': 1,
             'upper left': 2,
             'lower left': 3,
             'lower right': 4,
             'right': 5,
             'center left': 6,
             'center right': 7,
             'lower center': 8,
             'upper center': 9,
             'center': 10}
        Returns
        -------
        fig: Figure
            Matplotlib figure class
        ax: Axes or array of Axes
            ax can be either a single Axes object, or an array of Axes objects if more than one subplot was created.
            In this case only ONE subplot is created
        """
        anchored_text = AnchoredText(text, loc=loc)
        ax.add_artist(anchored_text)
        return fig, ax

    @staticmethod
    def add_normal_distribution_line(fig, ax, A, mu, sigma, fmt=None, label=None):
        """
        Add normal distribution as line
        Parameters
        ----------
        fig: Figure
            Matplotlib figure class
        ax: Axes or array of Axes
            ax can be either a single Axes object, or an array of Axes objects if more than one subplot was created.
            In this case only ONE subplot is created
        fmt: str, optional
            A format string, e.g. 'ro' for red circles. See the matplotlib section for a full description of the format
            strings. Format strings are just an abbreviation for quickly setting basic line properties. All of these and
            more can also be controlled by keyword arguments. This argument cannot be passed as keyword.
        label: str, optional
            Label associate with the data points
        A: dtype float
            Gaussian fit coefficient.
        mu: dtype float
            Mean value of the dataset.
        sigma: dtype float
            Standard deviation of the dataset.

        Returns
        -------
        fig: Figure
            Matplotlib figure class
        ax: Axes or array of Axes
            ax can be either a single Axes object, or an array of Axes objects if more than one subplot was created.
            In this case only ONE subplot is created
        """
        x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 1000)
        y = NumericalMethods.calculate_normal_distribution(x, A, mu, sigma)
        return Plotter.add_line(fig, ax, x, y, fmt=fmt, label=label)

    @staticmethod
    def add_control_limit_line(fig, ax, dataset, limit, fmt=None, label=None):
        """
        Add control limits as line
        Parameters
        ----------
        fig: Figure
            Matplotlib figure class
        ax: Axes or array of Axes
            ax can be either a single Axes object, or an array of Axes objects if more than one subplot was created.
            In this case only ONE subplot is created
        fmt: str, optional
            A format string, e.g. 'ro' for red circles. See the matplotlib section for a full description of the format
            strings. Format strings are just an abbreviation for quickly setting basic line properties. All of these and
            more can also be controlled by keyword arguments. This argument cannot be passed as keyword.
        label: str, optional
            Label associate with the data points
        dataset: dict of array of dtype float
            Input data in time dependent dataset format
        limit: dtype float
            Lower Control Limit (LCL), Upper Control Limit (UCL), Center Line (CL)

        Returns
        -------
        fig: Figure
            Matplotlib figure class
        ax: Axes or array of Axes
            ax can be either a single Axes object, or an array of Axes objects if more than one subplot was created.
            In this case only ONE subplot is created
        """
        x = [min(list(dataset.keys())), max(list(dataset.keys()))]
        return Plotter.add_line(fig, ax, x, [limit, limit], fmt=fmt, label=label)

    @staticmethod
    def add_normal_distribution_text_box(fig, ax, mu, sigma, N, p_value):
        """
        Add normal distribution text box
        Parameters
        ----------
        fig: Figure
            Matplotlib figure class
        ax: Axes or array of Axes
            ax can be either a single Axes object, or an array of Axes objects if more than one subplot was created.
            In this case only ONE subplot is created
        mu: dtype float
            Mean value of the dataset.
        sigma: dtype float
            Standard deviation of the dataset.
        N: int
            Number of samples
        p_value:
            P-value of the test

        Returns
        -------
        fig: Figure
            Matplotlib figure class
        ax: Axes or array of Axes
            ax can be either a single Axes object, or an array of Axes objects if more than one subplot was created.
            In this case only ONE subplot is created
        """

        text = '\n'.join(["\u03BC: {:.3f}".format(mu),
                          "\u03C3: {:.3f}".format(sigma),
                          "N: {:d}".format(N),
                          "p-value:  {:.3f}".format(p_value)])
        return Plotter.add_text_box(fig, ax, text)

    @staticmethod
    def add_quartile_text_box(fig, ax, minimum, first_quartile, median, third_quartile, maximum):
        """
        Add quartile text box
        Parameters
        ----------
        fig: Figure
            Matplotlib figure class
        ax: Axes or array of Axes
            ax can be either a single Axes object, or an array of Axes objects if more than one subplot was created.
            In this case only ONE subplot is created
        minimum: dtype float
            Minimum value of dataset. 0%
        first_quartile: dtype float
            1st quartile - 25%
        median: dtype float
            2nd quartile - 50%
        third_quartile: dtype float
            3rd quartile - 75%
        maximum: dtype float
            4th quartile - 100%

        Returns
        -------
        fig: Figure
            Matplotlib figure class
        ax: Axes or array of Axes
            ax can be either a single Axes object, or an array of Axes objects if more than one subplot was created.
            In this case only ONE subplot is created
        """

        text = '\n'.join(["Minimum: {:.3f}".format(minimum),
                          "1st quartile: {:.3f}".format(first_quartile),
                          "Median:  {:.3f}".format(median),
                          "3rd quartile: {:.3f}".format(third_quartile),
                          "Maximum: {:.3f}".format(maximum)])
        return Plotter.add_text_box(fig, ax, text)

    @staticmethod
    def show(fig, ax):
        """
        Show the plot
        Parameters
        ----------
        fig: Figure
            Matplotlib figure class
        ax: Axes or array of Axes
            ax can be either a single Axes object, or an array of Axes objects if more than one subplot was created.
            In this case only ONE subplot is created

        Returns
        -------
        fig: Figure
            Matplotlib figure class
        ax: Axes or array of Axes
            ax can be either a single Axes object, or an array of Axes objects if more than one subplot was created.
            In this case only ONE subplot is created

        """
        fig.tight_layout()
        plt.show()
        return fig, ax
