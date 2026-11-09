"""Static matplotlib figures for the report.

Style rules: one series per panel (no dual axes), thin 1.5px lines, recessive
grid, and text in neutral ink rather than series colour. Correlations use a
diverging blue-grey-red scale centred on zero.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # file output only; no display needed
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from statsmodels.tsa.stattools import acf  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = "#2a78d6"
SHADE = "#f0efec"
DIVERGING = LinearSegmentedColormap.from_list("blue_grey_red", ["#2a78d6", "#f0efec", "#e34948"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
    "axes.titlesize": 9, "axes.titleweight": "bold", "axes.labelsize": 8,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.family": "DejaVu Sans", "lines.linewidth": 1.5,
})


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def small_multiples(
    df: pd.DataFrame,
    path: Path,
    *,
    title: str,
    ncols: int = 3,
    zero_line: bool = False,
    shade: tuple[str, str, str] | None = None,
) -> Path:
    """One panel per column, shared x-axis. ``shade`` = (start, end, label) marks a period."""
    n = df.shape[1]
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 2.1 * nrows), sharex=True, squeeze=False)
    for ax, col in zip(axes.flat, df.columns):
        if shade:
            ax.axvspan(pd.Timestamp(shade[0]), pd.Timestamp(shade[1]), color=SHADE, zorder=0, lw=0)
        if zero_line:
            ax.axhline(0, color=AXIS, lw=0.8, zorder=1)
        ax.plot(df.index, df[col], color=SERIES, zorder=2)
        ax.set_title(col, loc="left")
    for ax in list(axes.flat)[n:]:
        ax.set_visible(False)
    subtitle = f"  (shaded: {shade[2]})" if shade else ""
    fig.suptitle(title + subtitle, x=0.01, ha="left", fontsize=11, color=INK, fontweight="bold")
    fig.tight_layout()
    return _save(fig, path)


def correlation_heatmap(df: pd.DataFrame, path: Path, *, title: str) -> Path:
    """Pearson correlation matrix with values printed in each cell."""
    corr = df.corr()
    k = len(corr)
    fig, ax = plt.subplots(figsize=(0.75 * k + 2.2, 0.65 * k + 1.2))
    im = ax.imshow(corr.to_numpy(), cmap=DIVERGING, vmin=-1, vmax=1)
    ax.set_xticks(range(k), corr.columns, rotation=40, ha="right")
    ax.set_yticks(range(k), corr.index)
    ax.grid(False)
    for i in range(k):
        for j in range(k):
            ax.text(j, i, f"{corr.iat[i, j]:.2f}", ha="center", va="center", fontsize=7, color=INK)
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=7, colors=MUTED)
    ax.set_title(title, loc="left", fontsize=10)
    fig.tight_layout()
    return _save(fig, path)


def acf_plot(series: pd.Series, path: Path, *, title: str, nlags: int = 24) -> Path:
    """Autocorrelation bars with a 95 % band (±1.96/√n) for 'no autocorrelation'."""
    x = series.dropna()
    values = acf(x, nlags=nlags, fft=True)[1:]
    band = 1.96 / np.sqrt(len(x))
    fig, ax = plt.subplots(figsize=(7, 2.6))
    ax.axhspan(-band, band, color=SHADE, lw=0, zorder=0)
    ax.axhline(0, color=AXIS, lw=0.8)
    ax.bar(range(1, nlags + 1), values, width=0.5, color=SERIES, zorder=2)
    ax.set_xlabel("lag (months)")
    ax.set_ylabel("autocorrelation")
    ax.set_title(f"{title}  (shaded: 95% band for zero autocorrelation)", loc="left", fontsize=10)
    fig.tight_layout()
    return _save(fig, path)
