import matplotlib.pyplot as plt
import pytest

from asalix.plotter import Plotter


@pytest.mark.parametrize('labels', [['first', 'second'], ('alpha', 'beta')])
def test_boxplot_preserves_custom_labels(labels):
    figure, axes = plt.subplots()
    try:
        result = Plotter.add_boxplot(figure, axes, [[1, 2, 3], [4, 5, 6]], labels)
        assert result == (figure, axes)
        assert [tick.get_text() for tick in axes.get_xticklabels()] == list(labels)
        figure.canvas.draw()
    finally:
        plt.close(figure)


def test_boxplot_preserves_default_blank_label():
    figure, axes = plt.subplots()
    try:
        Plotter.add_boxplot(figure, axes, [1, 2, 3])
        assert [tick.get_text() for tick in axes.get_xticklabels()] == ['']
        figure.canvas.draw()
    finally:
        plt.close(figure)
