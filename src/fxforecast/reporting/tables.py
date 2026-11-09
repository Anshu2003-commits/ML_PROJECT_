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
    """Index becomes the first column; floats are rounded to ``digits``."""
    index_name = df.index.name or ""
    header = [index_name, *map(str, df.columns)]
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] * len(header)) + "|",
    ]
    for idx, row in df.iterrows():
        cells = [str(idx), *(_fmt(v, digits) for v in row.tolist())]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
