"""Simple GUI controls for the proximity heatmap visualization."""

from matplotlib.widgets import Button


class ControlPanel:
    """Collection of buttons wired to callback functions.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to which the controls will be attached.
    callbacks : dict[str, callable]
        Mapping from button label to callback function.
    """

    def __init__(self, fig, callbacks):
        self.buttons = {}

        # Layout buttons vertically on the right side of the figure.
        width = 0.1
        height = 0.05
        left = 0.86
        bottom = 0.05
        padding = 0.01

        for i, (label, callback) in enumerate(callbacks.items()):
            ax = fig.add_axes([left, bottom + i * (height + padding), width, height])
            button = Button(ax, label)
            button.on_clicked(callback)
            self.buttons[label] = button

    def __getitem__(self, label):
        return self.buttons[label]
