from PySALESetup.functions import get_figure_from_ax, \
    _convert_input_to_fortran_strings
import matplotlib.pyplot as plt
import pytest


class TestGetFigureFromAx:
    def test_happy(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        fig2 = get_figure_from_ax(ax)
        assert fig2[1] is fig

    def test_no_ax_provided(self):
        results = get_figure_from_ax()
        assert len(results) == 2

    def test_fig_second_in_results(self):
        results = get_figure_from_ax()
        assert(isinstance(results[1], plt.Figure))

    def test_ax_first_in_results(self):
        results = get_figure_from_ax()
        assert(isinstance(results[0], plt.Axes))


class TestConvertToFortranStrings:
    @pytest.mark.parametrize('input_', [1, 1., '1'])
    def test_output_string(self, input_):
        out = _convert_input_to_fortran_strings({'a': input_})
        assert isinstance(out['a'], str)

    def test_float_in_standard_form(self):
        out = _convert_input_to_fortran_strings({'a': 1.})
        assert 'D' in out['a']

