from unittest import TestCase
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np

import asalix


class HistogramFiguresTest(TestCase):
    def setUp(self):
        self.sample = np.random.default_rng(42).normal(size=200)
        plt.close('all')
        self.addCleanup(plt.close, 'all')

    def test_numeric_histograms_do_not_register_figures(self):
        for fit in (False, True):
            with self.subTest(fit=fit):
                result = asalix.create_histogram(
                    self.sample, plot=False, normal_distribution_fitting=fit)
                np.testing.assert_array_equal(result.hist, np.histogram(self.sample, bins=10)[0])
                self.assertEqual(plt.get_fignums(), [])

    def test_numeric_histogram_does_not_modify_an_existing_figure(self):
        figure, axes = plt.subplots(num=73)
        axes.plot([0, 1], [1, 2])
        axes.set_title('Caller-owned figure')
        asalix.create_histogram(self.sample, plot=False, fig_number=73)
        self.assertEqual(figure.axes, [axes])
        self.assertEqual(axes.get_title(), 'Caller-owned figure')
        self.assertEqual(len(axes.lines), 1)

    def test_plotting_still_draws_and_shows_the_requested_histogram(self):
        with patch.object(asalix.Plotter, 'show') as show:
            result = asalix.create_histogram(self.sample, bins=8, plot=True, fig_number=74)
        show.assert_called_once()
        figure, axes = show.call_args.args
        self.assertEqual(figure.number, 74)
        self.assertEqual(axes.get_title(), 'Histogram')
        self.assertEqual(len(axes.patches), 8)
        np.testing.assert_array_equal(result.hist, np.histogram(self.sample, bins=8)[0])
