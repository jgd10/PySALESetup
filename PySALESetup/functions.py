import numbers
from typing import Optional, Tuple, Dict, Union
import matplotlib.pyplot as plt


def get_figure_from_ax(ax: Optional[plt.Axes] = None) -> Tuple[plt.Axes,
                                                               plt.Figure]:
    """Get the matplotlib figure from an Axes or create some if None.

    Parameters
    ----------
    ax : plt.Axes

    Returns
    -------
    (ax, fig) : Tuple[plt.Axes, plt.figure]
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal')
    else:
        fig = ax.get_figure()
    return ax, fig


def _convert_input_to_fortran_strings(
        input_values: Dict[str, Union[str, float, int]]
) -> Dict[str, str]:
    new = {}
    for key, value in input_values.items():
        if isinstance(value, numbers.Number) and \
                not isinstance(value, int):
            value = f'{value:.6e}'.replace('e', 'D')
        else:
            value = f'{value}'
        new[key] = value
    return new
