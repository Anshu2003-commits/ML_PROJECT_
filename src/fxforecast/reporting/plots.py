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


# Categorical slots in fixed order (validated: scripts/validate_palette.js). Two pairs sit in the
# 6-8 CVD band, so every series also gets its own line style / marker and a direct label.
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#e87ba4"]
LINESTYLES = ["-", "--", "-.", ":"]
MARKERS = ["o", "s", "^", "D"]


def forecast_paths(actual: pd.Series, forecasts: dict[str, pd.Series], path: Path, *, title: str) -> Path:
    """Actual series in ink plus up to four forecast series, each with its own colour and line style.

    Forecast lines end close together, so identity comes from the legend + line style, not end labels.
    """
    if len(forecasts) > len(CATEGORICAL):
        raise ValueError(f"At most {len(CATEGORICAL)} forecast series")
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(actual.index, actual, color=INK, lw=2.2, label="actual", zorder=5)
    for i, (name, s) in enumerate(forecasts.items()):
        ax.plot(s.index, s, color=CATEGORICAL[i], ls=LINESTYLES[i], lw=1.5, label=name, zorder=3)
    ax.set_ylabel("INR per USD")
    ax.legend(frameon=False, fontsize=7, ncol=len(forecasts) + 1, loc="upper left")
    ax.set_title(title, loc="left", fontsize=10)
    fig.tight_layout()
    return _save(fig, path)


def theil_u_dots(table: pd.DataFrame, path: Path, *, title: str, reference: dict[str, float] | None = None) -> Path:
    """Small multiples by column (e.g. horizon): one row per model, one dot per feature set.

    ``table`` index = (model, feature_set), columns = panels; values = Theil's U.
    A vertical line at 1 marks the random walk; ``reference`` adds a dashed line per panel (e.g. drift).
    """
    models = list(dict.fromkeys(table.index.get_level_values(0)))
    sets = list(dict.fromkeys(table.index.get_level_values(1)))
    fig, axes = plt.subplots(1, table.shape[1], figsize=(3.0 * table.shape[1] + 1, 0.42 * len(models) + 1.4),
                             sharey=True, squeeze=False)
    lo, hi = np.nanmin(table.to_numpy()), np.nanmax(table.to_numpy())
    for ax, col in zip(axes[0], table.columns):
        ax.axvline(1.0, color=AXIS, lw=1.2, zorder=1)
        if reference and col in reference:
            ax.axvline(reference[col], color=INK_2, lw=1, ls="--", zorder=1)
        for j, fs in enumerate(sets):
            ys, xs = [], []
            for i, m in enumerate(models):
                if (m, fs) in table.index and not np.isnan(table.loc[(m, fs), col]):
                    ys.append(i)
                    xs.append(table.loc[(m, fs), col])
            ax.scatter(xs, ys, s=46, color=CATEGORICAL[j], marker=MARKERS[j], label=fs, zorder=3,
                       edgecolors=SURFACE, linewidths=1.5)
        ax.set_title(str(col), loc="left")
        ax.set_xlim(lo - 0.03, hi + 0.03)
        ax.grid(axis="y", visible=False)
    axes[0][0].set_yticks(range(len(models)), models)
    axes[0][0].invert_yaxis()
    axes[0][-1].legend(frameon=False, fontsize=7, loc="lower right")
    note = "solid line: random walk (U = 1)" + ("; dashed: drift" if reference else "") + "; left of the line = better"
    fig.suptitle(f"{title}\n{note}", x=0.01, ha="left", fontsize=10, color=INK, fontweight="bold")
    fig.tight_layout()
    return _save(fig, path)


def ratio_heatmap(table: pd.DataFrame, path: Path, *, title: str, center: float = 1.0, span: float = 0.4) -> Path:
    """Heatmap of ratios (e.g. Theil's U) on a diverging scale centred at ``center``; values printed."""
    k_rows, k_cols = table.shape
    fig, ax = plt.subplots(figsize=(1.3 * k_cols + 3.2, 0.36 * k_rows + 1.3))
    im = ax.imshow(table.to_numpy(dtype=float), cmap=DIVERGING, vmin=center - span, vmax=center + span, aspect="auto")
    ax.set_xticks(range(k_cols), table.columns)
    ax.set_yticks(range(k_rows), [" / ".join(map(str, i)) if isinstance(i, tuple) else str(i) for i in table.index])
    ax.grid(False)
    for i in range(k_rows):
        for j in range(k_cols):
            ax.text(j, i, f"{table.iat[i, j]:.2f}", ha="center", va="center", fontsize=7, color=INK)
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=7, colors=MUTED)
    ax.set_title(title, loc="left", fontsize=10)
    fig.tight_layout()
    return _save(fig, path)


def grouped_bars(table: pd.DataFrame, path: Path, *, title: str, ylabel: str) -> Path:
    """Bars per index category, one bar per column (max 4 columns), values printed above bars."""
    k, m = table.shape
    if m > len(CATEGORICAL):
        raise ValueError(f"At most {len(CATEGORICAL)} columns")
    width = 0.8 / m
    fig, ax = plt.subplots(figsize=(1.4 * k + 2, 3.4))
    for j, col in enumerate(table.columns):
        xs = np.arange(k) + (j - (m - 1) / 2) * width
        vals = table[col].to_numpy(dtype=float)
        ax.bar(xs, vals, width=width - 0.03, color=CATEGORICAL[j], label=col, zorder=2,
               hatch=None if j == 0 else "//", edgecolor=SURFACE, linewidth=0)
        for x, v in zip(xs, vals):
            if not np.isnan(v):
                ax.text(x, v, f"{v:.2f}", ha="center", va="bottom", fontsize=7, color=INK_2)
    ax.set_xticks(range(k), table.index)
    ax.set_ylabel(ylabel)
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title(title, loc="left", fontsize=10)
    fig.tight_layout()
    return _save(fig, path)


def fold_diagram(folds, path: Path, *, title: str) -> Path:
    """One row per fold: training span (muted) and test span (series colour)."""
    fig, ax = plt.subplots(figsize=(8, 0.28 * len(folds) + 1.2))
    for i, f in enumerate(folds):
        y = len(folds) - i
        ax.barh(y, f.train[-1] - f.train[0], left=f.train[0], height=0.6, color=AXIS, lw=0)
        ax.barh(y, (f.test[-1] - f.test[0]) + pd.Timedelta(days=28), left=f.test[0], height=0.6,
                color=SERIES, lw=0)
    ax.set_yticks([len(folds) - i for i in range(len(folds))],
                  [f"refit {f.refit_date:%Y-%m}" for f in folds])
    ax.grid(axis="y", visible=False)
    ax.set_title(f"{title}  (grey: train origins, blue: test origins)", loc="left", fontsize=10)
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
