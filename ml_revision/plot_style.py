"""Shared publication typography; does not modify scientific results."""
import matplotlib as mpl
from matplotlib import font_manager

font_manager.findfont('Arial', fallback_to_default=False)
mpl.rcParams.update({
    'font.family': 'Arial',
    'mathtext.fontset': 'custom',
    'mathtext.rm': 'Arial',
    'mathtext.it': 'Arial:italic',
    'mathtext.bf': 'Arial:bold',
    'svg.fonttype': 'none',
    'pdf.fonttype': 42,
})
