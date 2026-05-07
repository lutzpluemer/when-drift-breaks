"""
teiresias.data
==============

Data-loading utilities with three-tier fallback strategy.

For each data source (SPY, CBOE Skew Index, S&P sector ETFs) we try, in
order:

1. **Local Parquet cache** -- if a cached file is present at the
   configured path, load from there (fastest, no network, no credentials).
2. **EODHD live fetch** -- if an EODHD API key is provided, fetch fresh
   adjusted-close data and write it to the cache.  This is the path used
   for the manuscript's main results.
3. **Yahoo Finance via yfinance** -- public fallback that does not
   require a subscription.  Used for the public companion repository
   so reviewers without an EODHD subscription can still reproduce the
   methodology end-to-end.

Yahoo's ``Adj Close`` and EODHD's ``adjusted_close`` differ by a small
amount (typically 0.01-0.05 % around dividend dates) due to different
adjustment conventions.  Numerical results are not bit-identical between
the two sources, but the qualitative findings of the *When Drift Breaks*
paper hold under both.  See ``data/PROVENANCE.md`` (when present).

Configuration
-------------

``EODHD_API_KEY`` is read from the environment variable of the same
name.  The Yahoo path requires the ``yfinance`` package
(``pip install yfinance``).

Symbols
-------

* SPY:     ``SPY.US``    (EODHD) / ``SPY``     (Yahoo)
* SKEW:    ``SKEW.INDX`` (EODHD) / ``^SKEW``   (Yahoo)
* Sectors: ``XLK.US`` ... (EODHD) / ``XLK`` ... (Yahoo)
"""

from __future__ import annotations

import io
import os
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Sector universe used in Phase Shape+Skew+Spectral
# ---------------------------------------------------------------------------

SECTOR_TICKERS: dict[str, str] = {
    "XLK": "Technology",
    "XLF": "Financials",
    "XLV": "Health Care",
    "XLE": "Energy",
    "XLY": "Consumer Discretionary",
    "XLP": "Consumer Staples",
    "XLI": "Industrials",
    "XLB": "Materials",
    "XLU": "Utilities",
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ensure_dir(path: str | os.PathLike) -> None:
    """Create the parent directory of ``path`` if it does not exist."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _read_cache(cache_path: str | None) -> pd.DataFrame | None:
    """Load a Parquet cache file if it exists, else None."""
    if cache_path and Path(cache_path).exists():
        return pd.read_parquet(cache_path)
    return None


def _write_cache(df: pd.DataFrame, cache_path: str | None) -> None:
    """Write a DataFrame to Parquet cache if a path is given."""
    if cache_path:
        _ensure_dir(cache_path)
        df.to_parquet(cache_path)


def _eodhd_get(
    symbol: str,
    api_key: str,
    start: str = "1990-01-01",
    timeout: int = 60,
) -> pd.DataFrame:
    """Fetch one symbol from EODHD's daily endpoint as a DataFrame."""
    import requests

    url = (
        f"https://eodhd.com/api/eod/{symbol}"
        f"?from={start}&period=d&fmt=csv&api_token={api_key}"
    )
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text))
    df.columns = [c.lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    return df


def _yahoo_get(
    symbol: str,
    start: str = "1990-01-01",
) -> pd.Series:
    """Fetch one symbol's adjusted close from Yahoo Finance via yfinance."""
    try:
        import yfinance as yf
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "yfinance is not installed.  Install it with "
            "`pip install yfinance` to use the Yahoo Finance fallback."
        ) from e

    ticker = yf.Ticker(symbol)
    hist = ticker.history(start=start, auto_adjust=False)
    if hist.empty:
        raise RuntimeError(f"Yahoo returned empty history for {symbol!r}")
    # yfinance returns a tz-aware index; strip tz for consistency with EODHD
    if hist.index.tz is not None:
        hist.index = hist.index.tz_localize(None)
    return hist["Adj Close"].rename("adjusted_close")


# ---------------------------------------------------------------------------
# SPY
# ---------------------------------------------------------------------------

def load_spy(
    cache_path: str | None = None,
    source: str = "auto",
    api_key: str | None = None,
    start: str = "1990-01-01",
) -> pd.Series:
    """
    Load SPY adjusted close prices.

    Parameters
    ----------
    cache_path : str, optional
        Path to a Parquet cache file.  If the file exists, it is loaded
        and returned without network access.  If a fresh download
        succeeds, the result is also written here.
    source : {'auto', 'eodhd', 'yahoo'}
        Which live source to use if the cache is empty or absent.

        * ``'auto'`` (default) -- prefer EODHD if an API key is available,
          else fall back to Yahoo.
        * ``'eodhd'`` -- require EODHD; raise if no API key.
        * ``'yahoo'`` -- use Yahoo Finance only.
    api_key : str, optional
        EODHD API key.  If ``None`` and EODHD is requested, the function
        reads ``os.environ['EODHD_API_KEY']``.
    start : str
        ISO-format start date for the download.

    Returns
    -------
    pd.Series
        Adjusted-close prices indexed by date, named ``adjusted_close``.
    """
    cached = _read_cache(cache_path)
    if cached is not None:
        # Cached file may be a Series (one column) or a DataFrame
        if isinstance(cached, pd.DataFrame):
            col = "adjusted_close" if "adjusted_close" in cached.columns else cached.columns[0]
            return cached[col].rename("adjusted_close")
        return cached.rename("adjusted_close")

    src = _resolve_source(source, api_key)
    if src == "eodhd":
        key = api_key or os.environ.get("EODHD_API_KEY")
        df = _eodhd_get("SPY.US", api_key=key, start=start)
        series = df["adjusted_close"].rename("adjusted_close")
    else:  # yahoo
        series = _yahoo_get("SPY", start=start)

    _write_cache(series.to_frame("adjusted_close"), cache_path)
    return series


# ---------------------------------------------------------------------------
# SKEW
# ---------------------------------------------------------------------------

def load_skew(
    cache_path: str | None = None,
    source: str = "auto",
    api_key: str | None = None,
    start: str = "1990-01-01",
) -> pd.Series:
    """
    Load the CBOE Skew Index.

    See :func:`load_spy` for the parameter semantics.  Yahoo's ticker for
    SKEW is ``^SKEW``; EODHD uses ``SKEW.INDX``.
    """
    cached = _read_cache(cache_path)
    if cached is not None:
        if isinstance(cached, pd.DataFrame):
            col = "adjusted_close" if "adjusted_close" in cached.columns else cached.columns[0]
            return cached[col].rename("skew")
        return cached.rename("skew")

    src = _resolve_source(source, api_key)
    if src == "eodhd":
        key = api_key or os.environ.get("EODHD_API_KEY")
        df = _eodhd_get("SKEW.INDX", api_key=key, start=start)
        # SKEW index uses the close column; adjusted_close may equal close
        col = "adjusted_close" if "adjusted_close" in df.columns else "close"
        series = df[col].rename("skew")
    else:  # yahoo
        series = _yahoo_get("^SKEW", start=start).rename("skew")

    _write_cache(series.to_frame("skew"), cache_path)
    return series


# ---------------------------------------------------------------------------
# Sector ETFs
# ---------------------------------------------------------------------------

def load_sectors(
    cache_path: str | None = None,
    source: str = "auto",
    api_key: str | None = None,
    start: str = "1995-01-01",
    tickers: dict[str, str] | None = None,
) -> pd.DataFrame:
    """
    Load adjusted-close prices for the nine S&P sector ETFs.

    Parameters
    ----------
    cache_path : str, optional
        Parquet cache path.
    source, api_key, start
        See :func:`load_spy`.
    tickers : dict, optional
        Override the default sector universe.  Keys are tickers, values
        are descriptive names.  Defaults to :data:`SECTOR_TICKERS`.

    Returns
    -------
    pd.DataFrame
        One column per ticker (ticker symbol as column name), indexed by
        date.  Missing values appear before each ticker's inception date.
    """
    cached = _read_cache(cache_path)
    if cached is not None:
        return cached

    if tickers is None:
        tickers = SECTOR_TICKERS

    src = _resolve_source(source, api_key)
    frames: dict[str, pd.Series] = {}

    if src == "eodhd":
        key = api_key or os.environ.get("EODHD_API_KEY")
        for ticker in tickers:
            df = _eodhd_get(f"{ticker}.US", api_key=key, start=start)
            frames[ticker] = df["adjusted_close"]
    else:  # yahoo
        for ticker in tickers:
            frames[ticker] = _yahoo_get(ticker, start=start)

    out = pd.DataFrame(frames)
    _write_cache(out, cache_path)
    return out


# ---------------------------------------------------------------------------
# Source resolution
# ---------------------------------------------------------------------------

def _resolve_source(source: str, api_key: str | None) -> str:
    """Decide between 'eodhd' and 'yahoo' given a user preference."""
    if source == "eodhd":
        key = api_key or os.environ.get("EODHD_API_KEY")
        if not key:
            raise RuntimeError(
                "source='eodhd' requested but no API key found.  "
                "Set EODHD_API_KEY or pass api_key= explicitly."
            )
        return "eodhd"
    if source == "yahoo":
        return "yahoo"
    if source == "auto":
        key = api_key or os.environ.get("EODHD_API_KEY")
        return "eodhd" if key else "yahoo"
    raise ValueError(
        f"Unknown source {source!r}.  Choose from 'auto', 'eodhd', 'yahoo'."
    )
