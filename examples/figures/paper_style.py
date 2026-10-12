"""
paper_style.py — shared matplotlib style for all paper figures.

Single source of truth for fonts, colors, sizes, and helpers so every
figure in the NeurIPS submission looks like it came out of the same
typesetting system.
"""

from __future__ import annotations

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


# Column widths in inches for the NeurIPS 2026 single-column layout
# (textwidth ≈ 5.5"; we leave a little slack for \linewidth=0.95).
COL_FULL = 5.5
COL_HALF = 2.7
ROW_SHORT = 2.0
ROW_TALL = 3.6


# Consistent palette across all figures.
PALETTE = {
    "seq":         "#1f77b4",   # sequential
    "par":         "#d62728",   # parallel (DEER)
    "par_alt":     "#9467bd",   # second parallel variant (full-Jac, learned, etc.)
    "bpf":         "#7f7f7f",   # bootstrap PF
    "ffbsi":       "#ff7f0e",   # FFBSi
    "twisted_seq": "#2ca02c",   # twisted sequential reference
    "truth":       "black",
    "ref":         "#888888",   # theoretical reference dashes
    "twseq":       "#1f77b4",
    "standard":    "#7f7f7f",
}

MARKERS = {
    "seq":         "^",
    "par":         "*",
    "bpf":         "o",
    "ffbsi":       "s",
    "twisted_seq": "^",
    "twisted_par": "*",
}


def apply():
    """Apply the shared rcParams.  Idempotent — safe to call repeatedly."""
    mpl.rcParams.update({
        # Typography — match the NeurIPS LaTeX template's serif body text
        "font.family":      "serif",
        "font.serif":       ["STIXGeneral", "Liberation Serif", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size":        9,
        "axes.titlesize":   10,
        "axes.labelsize":   9,
        "xtick.labelsize":  8,
        "ytick.labelsize":  8,
        "legend.fontsize":  8,
        "legend.frameon":   True,
        "legend.framealpha": 0.85,
        "legend.edgecolor": "0.6",
        # Axes
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.linewidth":    0.8,
        "axes.grid":         True,
        "grid.linewidth":    0.4,
        "grid.alpha":        0.35,
        "grid.color":        "#cccccc",
        "axes.axisbelow":    True,
        # Lines & markers
        "lines.linewidth":   1.4,
        "lines.markersize":  5,
        "lines.markeredgewidth": 0.6,
        # Saving
        "savefig.dpi":       250,
        "savefig.bbox":      "tight",
        "savefig.pad_inches": 0.03,
        "figure.dpi":        110,
        # Make small ticks visible
        "xtick.major.size":  3.0,
        "ytick.major.size":  3.0,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
    })


def add_power_reference(ax, x, exponent, *, c=1.0, label=None, xy_anchor=None,
                         color=None, ls="--", lw=1.0, alpha=0.7):
    """Overlay a y = c * x^exponent reference dashed line on a log-log axis.

    `xy_anchor`: optional (x0, y0) point through which the line should pass.
    If given, c is computed so the line passes through that anchor.
    """
    x = np.asarray(x, dtype=float)
    if xy_anchor is not None:
        x0, y0 = xy_anchor
        c = y0 / (x0 ** exponent)
    y = c * x ** exponent
    color = color or PALETTE["ref"]
    ax.plot(x, y, ls=ls, lw=lw, color=color, alpha=alpha, label=label,
            zorder=1)


def add_log_reference(ax, x, *, c=1.0, label=None, xy_anchor=None,
                       color=None, ls=":", lw=1.0, alpha=0.7):
    """Overlay a y = c * log(x) reference for "flat / log" parallel time."""
    x = np.asarray(x, dtype=float)
    if xy_anchor is not None:
        x0, y0 = xy_anchor
        c = y0 / np.log(x0)
    y = c * np.log(x)
    color = color or PALETTE["ref"]
    ax.plot(x, y, ls=ls, lw=lw, color=color, alpha=alpha, label=label,
            zorder=1)


def save_both(fig, out, **kwargs):
    """Save ``fig`` as both PNG and SVG.

    ``out`` may end in ``.png`` or ``.svg``; both files are written next to
    each other so paper figures can be edited in vector tools while still
    embedding the rasterized version.
    """
    base, ext = os.path.splitext(out)
    if ext.lower() not in (".png", ".svg"):
        base = out
    fig.savefig(base + ".png", **kwargs)
    fig.savefig(base + ".svg", **kwargs)


def annotate_box(ax, text, *, loc="upper right", pad=0.02):
    """Drop a small annotation in a corner of an axes."""
    locs = {
        "upper right": (1 - pad, 1 - pad, "right", "top"),
        "upper left":  (pad,     1 - pad, "left",  "top"),
        "lower right": (1 - pad, pad,     "right", "bottom"),
        "lower left":  (pad,     pad,     "left",  "bottom"),
    }
    x, y, ha, va = locs[loc]
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va,
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                      ec="0.6", lw=0.6, alpha=0.9))
