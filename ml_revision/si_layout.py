"""Legible joint-panel labels at the SI's 6.5-inch figure width."""
import textwrap
from matplotlib.ticker import MaxNLocator


def style_joint(top, ax, label, *, dense=False):
    size=14.5 if dense else 14
    for axis in (ax.xaxis, ax.yaxis):
        axis.label.set_text(textwrap.fill(axis.label.get_text(),width=23,
                                         break_long_words=False,break_on_hyphens=False))
        axis.label.set_size(size)
        axis.labelpad=5
        axis.set_major_locator(MaxNLocator(4))
    ax.tick_params(labelsize=13.5 if dense else 13,pad=3)
    top.set_title(label,fontsize=14,pad=5,fontweight='bold')
