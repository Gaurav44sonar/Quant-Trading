"""
features/core.py
=================
Cross-sectional math primitives.

Every function here operates on a DataFrame where:
    rows    = timestamps  (bars)
    columns = tickers

All operations are vectorised with numpy — no Python loops.
These are the building blocks every feature layer uses.

Citadel design principle: separate the math from the economics.
Each primitive has ONE job and is unit-tested independently.
"""

from __future__ import annotations
import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Cross-sectional normalisation
# ─────────────────────────────────────────────────────────────────────────────

def cs_zscore(df: pd.DataFrame, clip: float = 3.0) -> pd.DataFrame:
    """
    Cross-sectional z-score at every bar.

    z(i,t) = (x(i,t) - μ_cs(t)) / σ_cs(t)

    Parameters
    ----------
    clip : winsorise at ±clip after z-scoring (default 3.0)

    Returns same shape as input. NaN tickers excluded from mean/std.
    """
    mu  = df.mean(axis=1)
    sig = df.std(axis=1).replace(0.0, np.nan)
    z   = df.sub(mu, axis=0).div(sig, axis=0)
    return z.clip(-clip, clip)


def cs_rank(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-sectional rank scaled to [-1, +1] at every bar.

    rank = -1 → lowest value (buy signal for reversal)
    rank = +1 → highest value (sell signal for reversal)

    Uses average rank for ties. NaN-safe.
    """
    ranked = df.rank(axis=1, method="average", na_option="keep")
    n      = df.notna().sum(axis=1).replace(0, np.nan)
    return (ranked.sub(1, axis=0).div(n - 1, axis=0) * 2 - 1).clip(-1, 1)


def cs_winsorise(df: pd.DataFrame, lower: float = 0.01,
                 upper: float = 0.99) -> pd.DataFrame:
    """
    Cross-sectional percentile winsorisation at every bar.
    Clips values outside [lower, upper] quantile at each timestamp.
    More robust than fixed-sigma clipping when distributions are heavy-tailed.
    """
    lo = df.quantile(lower, axis=1)
    hi = df.quantile(upper, axis=1)
    return df.clip(lo, hi, axis=0)


def cs_demean(df: pd.DataFrame) -> pd.DataFrame:
    """Subtract cross-sectional mean at every bar (dollar-neutral prep)."""
    return df.sub(df.mean(axis=1), axis=0)


def cs_neutralise(df: pd.DataFrame,
                  factor: pd.DataFrame) -> pd.DataFrame:
    """
    Remove a common factor from df via cross-sectional OLS at each bar.

    Usage: neutralise returns against market return, sector return, etc.
    Equivalent to computing residuals of:  df ~ factor   (no intercept)

    Parameters
    ----------
    df     : signals or returns  [T × N]
    factor : single factor       [T × N]  (e.g. market return broadcast)
    """
    # beta(t) = Σ df(i,t)*factor(i,t) / Σ factor(i,t)^2   (cross-sectional)
    num   = (df * factor).sum(axis=1)
    denom = (factor ** 2).sum(axis=1).replace(0.0, np.nan)
    beta  = num / denom
    return df.sub(beta.values[:, None] * factor.values, fill_value=np.nan)


# ─────────────────────────────────────────────────────────────────────────────
# Rolling statistics (time-series per ticker, vectorised across all tickers)
# ─────────────────────────────────────────────────────────────────────────────

def rolling_mean(df: pd.DataFrame, window: int,
                 min_periods: int | None = None) -> pd.DataFrame:
    mp = min_periods if min_periods is not None else max(1, window // 4)
    return df.rolling(window, min_periods=mp).mean()


def rolling_std(df: pd.DataFrame, window: int,
                min_periods: int | None = None) -> pd.DataFrame:
    mp = min_periods if min_periods is not None else max(2, window // 4)
    return df.rolling(window, min_periods=mp).std()


def rolling_median(df: pd.DataFrame, window: int,
                   min_periods: int | None = None) -> pd.DataFrame:
    mp = min_periods if min_periods is not None else max(1, window // 4)
    return df.rolling(window, min_periods=mp).median()


def rolling_mad(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Rolling Median Absolute Deviation × 1.4826.
    Robust scale estimator: consistent with std for normal distributions
    but ~5× more resistant to outliers.
    """
    def _mad_vec(x: np.ndarray) -> float:
        m = np.nanmedian(x)
        return np.nanmedian(np.abs(x - m)) * 1.4826

    mp = max(3, window // 4)
    return df.rolling(window, min_periods=mp).apply(_mad_vec, raw=True)


def ewm_std(df: pd.DataFrame, halflife: int) -> pd.DataFrame:
    """
    Exponentially-weighted standard deviation.
    halflife = N bars → weight decays to 0.5 after N bars.
    Faster to adapt to volatility regime changes than rolling std.
    """
    return df.ewm(halflife=halflife, min_periods=max(2, halflife // 2)).std()


def ewm_mean(df: pd.DataFrame, halflife: int) -> pd.DataFrame:
    return df.ewm(halflife=halflife, min_periods=max(1, halflife // 2)).mean()


# ─────────────────────────────────────────────────────────────────────────────
# Session utilities
# ─────────────────────────────────────────────────────────────────────────────

def session_cumsum(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cumulative sum that resets at the start of each trading session.
    Used for: session volume, session PnL, session bar count.
    """
    dates = df.index.normalize()
    return df.groupby(dates).cumsum()


def session_bar_index(index: pd.DatetimeIndex) -> pd.Series:
    """
    Integer bar position within the current session (0-based).
    Bar 0 = first bar of the session (e.g. 09:30 for 15-min bars).
    """
    dates = index.normalize()
    counts = pd.Series(index=index, dtype=int)
    for date, grp in pd.Series(index=index).groupby(dates):
        counts.loc[grp.index] = range(len(grp))
    return counts


def session_fraction(index: pd.DatetimeIndex,
                     bars_per_session: int = 26) -> pd.Series:
    """
    Fraction of session elapsed: 0.0 at open, 1.0 at close.
    Used to weight signals that strengthen/weaken through the day (e.g. VWAP).
    """
    bar_idx = session_bar_index(index)
    return bar_idx / (bars_per_session - 1)


def is_first_bar(index: pd.DatetimeIndex) -> pd.Series:
    """Boolean Series: True for the first bar of each session."""
    dates = index.normalize()
    first = pd.Series(False, index=index)
    for _, grp in pd.Series(index=index).groupby(dates):
        first.iloc[first.index.get_loc(grp.index[0])] = True
    return first


def is_last_bar(index: pd.DatetimeIndex,
                bars_per_session: int = 26) -> pd.Series:
    """Boolean Series: True for the last bar of each session."""
    bar_idx = session_bar_index(index)
    return pd.Series(bar_idx >= bars_per_session - 1, index=index)


# ─────────────────────────────────────────────────────────────────────────────
# ATR (Average True Range) — used by multiple features
# ─────────────────────────────────────────────────────────────────────────────

def true_range(high: pd.DataFrame, low: pd.DataFrame,
               close: pd.DataFrame) -> pd.DataFrame:
    """
    True Range = max(H-L, |H-C_prev|, |L-C_prev|)
    This is the correct volatility measure for bars with gaps.
    """
    prev_close = close.shift(1)
    hl  = high - low
    hcp = (high - prev_close).abs()
    lcp = (low  - prev_close).abs()
    return pd.concat([hl, hcp, lcp], axis=0).groupby(level=0).max()


def atr(high: pd.DataFrame, low: pd.DataFrame,
        close: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Average True Range over rolling window."""
    tr = true_range(high, low, close)
    return rolling_mean(tr, window)


def atr_pct(high: pd.DataFrame, low: pd.DataFrame,
            close: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """ATR as percentage of close — makes it comparable across price levels."""
    return atr(high, low, close, window) / close.replace(0.0, np.nan)
