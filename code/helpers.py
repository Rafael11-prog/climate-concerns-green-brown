# Shared functions for the notebooks of this project (Requirement 4.9).
# Drafted with Claude Code (Anthropic) and reviewed by the author.

from pathlib import Path

import pandas as pd

# Paths relative to the repository root, whether the notebook runs from code/ or from the root
ROOT = Path.cwd().parent if Path.cwd().name == "code" else Path.cwd()
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
RES = ROOT / "results"
FIG = ROOT / "figures"


def read_dated(path):
    """Read a CSV whose first column is a date, used as the index."""
    return pd.read_csv(path, index_col=0, parse_dates=True)


def read_processed(name):
    """Read a file of data/processed with its date index."""
    return read_dated(PROC / name)


def load_universe():
    """The stock universe (indexed by ticker) and the lists of green and brown European tickers."""
    universe = pd.read_csv(RAW / "universe.csv").set_index("ticker")
    green = universe.index[universe["group"] == "green_eu"].tolist()
    brown = universe.index[universe["group"] == "brown_eu"].tolist()
    return universe, green, brown


def side_return(rets, members, min_stocks):
    """Equal-weighted daily return of the members (missing if fewer than min_stocks), and the number of stocks."""
    r = rets[members]
    n = r.notna().sum(axis=1)
    return r.mean(axis=1).where(n >= min_stocks), n


def green_minus_brown(rets, green, brown, min_stocks):
    """Green and brown portfolio returns, the green-minus-brown spread, and the number of stocks per side."""
    g, ng = side_return(rets, green, min_stocks)
    b, nb = side_return(rets, brown, min_stocks)
    return g, b, g - b, ng, nb


def delayed_stocks(entry, entry_col):
    """Stocks whose liquidity entry year (entry[entry_col]) is later than the year of their first price."""
    first_year = pd.to_datetime(entry["first_price"]).dt.year
    return entry.index[entry[entry_col].astype(float) > first_year]
