"""Render DataFrames as Markdown tables (GitHub-flavoured) without extra dependencies."""
from __future__ import annotations

import numbers

import pandas as pd


def _fmt(value, digits: int) -> str:
    if isinstance(value, (bool,)) or value is None:
        return str(value)
    if isinstance(value, numbers.Integral):
        return str(value)
    if isinstance(value, numbers.Real):
        if pd.isna(value):
            return ""
        if value != 0 and abs(value) < 10 ** -digits:
            return f"{value:.1e}"
        return f"{value:.{digits}f}"
    return str(value)


def to_markdown(df: pd.DataFrame, digits: int = 3) -> str:
    """Index becomes the first column(s); floats are rounded to ``digits``, ints stay ints."""
    index_names = [n or "" for n in df.index.names]
    header = [*index_names, *map(str, df.columns)]
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] * len(header)) + "|",
    ]
    for idx, *values in df.itertuples(name=None):   # itertuples keeps each column's dtype
        keys = idx if isinstance(idx, tuple) else (idx,)
        cells = [*map(str, keys), *(_fmt(v, digits) for v in values)]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
