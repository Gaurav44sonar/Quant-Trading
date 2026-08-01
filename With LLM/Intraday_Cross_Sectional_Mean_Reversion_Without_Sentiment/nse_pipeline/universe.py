"""
nse_pipeline/universe.py
========================
Dynamic universe construction for NASDAQ intraday mean reversion.

Selects 180 volatile + 120 non-volatile stocks from a seed pool
based on trailing ATR% (Average True Range as % of price).

Algorithm:
  1. Start with a seed pool (~500 liquid NASDAQ-listed stocks)
  2. Fetch 6 months of daily OHLCV for all
  3. Apply filters: ADTV ≥ $1M, Price ≥ $5, History ≥ 60 days
  4. Compute ATR% = ATR(20) / Close for each stock
  5. Rank by ATR%
  6. Volatile 180 = Top-180 by ATR% (highest volatility)
  7. Non-Volatile 120 = Bottom-120 by ATR% (lowest volatility)

Usage
-----
    from nse_pipeline.universe import UniverseBuilder
    
    builder = UniverseBuilder()
    volatile, nonvolatile = builder.select()
    all_tickers = volatile + nonvolatile  # 300 stocks
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Seed pool: ~500 liquid NASDAQ-listed stocks
# These are screened dynamically — the final 300 depend on ATR% ranking.
# ─────────────────────────────────────────────────────────────────────────────

SEED_POOL = [
    # ══════════════════════════════════════════════════════════════════════
    # MEGA-CAP / NASDAQ-100 CORE (updated - only active NASDAQ-listed)
    # ══════════════════════════════════════════════════════════════════════
    
    # ── Big Tech / FAANG+ ─────────────────────────────────────────────
    "AAPL", "MSFT", "AMZN", "GOOGL", "GOOG", "META", "NVDA", "TSLA",
    "AVGO", "ADBE", "NFLX", "AMD", "INTC", "QCOM", "TXN",
    "CSCO", "AMAT", "MU", "LRCX", "KLAC", "MRVL",
    "SNPS", "CDNS", "ADI", "NXPI", "MCHP", "ON", "SWKS", "MPWR",
    
    # ── Internet / Software ───────────────────────────────────────────
    "PANW", "CRWD", "FTNT", "ZS", "DDOG", "MDB",
    "WDAY", "TEAM", "OKTA", "CDNS", "TTD", "ROKU", "ZM", "DOCU",
    "MNDY", "DKNG", "ABNB", "DASH", "LYFT",
    
    # ── Biotech / Pharma ──────────────────────────────────────────────
    "AMGN", "GILD", "REGN", "VRTX", "BIIB", "ILMN", "MRNA", "BNTX",
    "ALNY", "BMRN", "INCY", "RARE", "NBIX", "PCVX",
    "ARGX", "SRPT", "IONS", "UTHR", "LEGN", "HALO", "RYTM",
    
    # ── Healthcare / MedTech ──────────────────────────────────────────
    "ISRG", "DXCM", "IDXX", "ALGN", "TECH", "NTRA",
    "PODD", "AZTA",
    
    # ── Consumer / Retail ─────────────────────────────────────────────
    "AMZN", "COST", "PEP", "SBUX", "MNST", "KDP", "MDLZ", "KHC",
    "DLTR", "ROST", "ORLY", "POOL", "TSCO", "ULTA",
    "LULU", "CPRT", "EBAY", "MELI", "BKNG", "EXPE", "TCOM",
    "MAR", "WYNN",
    
    # ── EV / Clean Energy ────────────────────────────────────────────
    "RIVN", "LCID", "ENPH", "SEDG", "FSLR",
    
    # ── Telecom / Media ──────────────────────────────────────────────
    "CMCSA", "CHTR", "TMUS", "FOXA", "FOX", "WBD", "NWSA",
    "EA", "TTWO",
    
    # ── Finance (NASDAQ-listed) ──────────────────────────────────────
    "PYPL", "COIN", "HOOD", "SOFI", "AFRM", "UPST",
    "NDAQ", "CME", "MKTX",
    "CINF", "ERIE",
    
    # ══════════════════════════════════════════════════════════════════════
    # MID-CAP NASDAQ (updated - only active NASDAQ-listed)
    # ══════════════════════════════════════════════════════════════════════
    
    # ── Software / SaaS ──────────────────────────────────────────────
    "APPN", "BRZE", "GTLB", "QLYS", "TENB",
    "RPD", "VRNS", "SAIL", "FRSH", "ALRM", "EVCM", "PRGS", "MANH",
    "NCNO", "ALKT", "CVLT", "LITE", "VIAV",
    
    # ── Semiconductor Extended ───────────────────────────────────────
    "DIOD", "SLAB", "RMBS", "FORM", "CRUS", "SYNA", "SMTC",
    "POWI", "ALGM", "ACLS", "AMKR", "CEVA", "SITM", "AMBA",
    "MTSI", "OLED", "MKSI", "LSCC", "IPGP",
    
    # ── Biotech Extended ─────────────────────────────────────────────
    "EXEL", "CORT", "PTCT", "RCKT", "APLS", "KRYS", "INSM",
    "CRNX", "VERA", "XNCR", "IMVT", "TGTX", "FOLD", "VCEL",
    "RVMD", "ARWR", "CPRX", "IRTC", "NVAX", "BBIO",
    "RXRX", "TVTX", "KROS", "RNA", "DAWN", "GERN", "IOVA",
    "PCVX", "CYTK", "SRRK", "PRTA", "ACAD",
    
    # ── Healthcare Services ──────────────────────────────────────────
    "HSIC", "OMCL", "PGNY", "OFIX", "TNDM", "ATRC", "CORT",
    
    # ── Consumer / E-Commerce ────────────────────────────────────────
    "REAL", "CARG", "OPEN", "ZG", "Z", "GRPN", "ANGI", "MTCH", "BMBL",
    "TRIP", "ABNB",
    
    # ── Industrials / Clean Tech ─────────────────────────────────────
    "PLUG", "BLDP", "RUN", "ARRY", "MAXN", "SPWR",
    "EVGO", "BLNK", "WKHS",
    
    # ── Fintech / Payments ───────────────────────────────────────────
    "RPAY", "PAYO", "FLYW", "RELY", "DLO", "BTBT", "MARA", "RIOT", "CLSK", "HUT",
    
    # ── Media / Entertainment ────────────────────────────────────────
    "SIRI", "IHRT", "DUOL", "UDMY",
    
    # ── Food / Beverage ──────────────────────────────────────────────
    "CELH", "FIZZ", "FRPT", "SMPL", "HAIN", "BYND", "LNTH",
    
    # ══════════════════════════════════════════════════════════════════════
    # SMALL-CAP / HIGH-VOLATILITY NASDAQ (updated - only active NASDAQ-listed)
    # ══════════════════════════════════════════════════════════════════════
    
    # ── Speculative Tech / AI ────────────────────────────────────────
    "PLTR", "PRCT", "SOUN", "GFAI", "UPST", "CLOV",
    "RGTI", "QUBT", "ARQQ",
    
    # ── Speculative Biotech ──────────────────────────────────────────
    "MVIS", "BNGO", "SNDL", "TLRY", "CGC", "ACB", "CRON", "GRFS",
    "VUZI", "WIMI", "INO", "OCGN", "AQST", "TARS", "CMPS", "ATAI", "CLOV",
    
    # ── Meme / Retail Favorites ──────────────────────────────────────
    "SOFI", "PLTR", "HOOD", "RIVN", "LCID", "LI",
    
    # ── Crypto / Blockchain ──────────────────────────────────────────
    "MSTR", "COIN", "MARA", "RIOT", "CLSK", "HUT", "CIFR",
    "BTBT", "BTDR", "IREN",
    
    # ── China ADRs (NASDAQ-listed) ───────────────────────────────────
    "PDD", "JD", "BIDU", "NTES", "BILI", "IQ",
    "FUTU", "TIGR", "VNET", "WB", "GDS", "KC", "QFIN", "LX", "NIU",
    
    # ── SPACs / Recent IPOs (high vol) ───────────────────────────────
    "SMCI", "ARM", "CART", "DUOL", "GRAB",
    
    # ── REITs / Other (NASDAQ-listed) ────────────────────────────────
    "EQIX", "SBAC", "LAMR",
    
    # ── Transport / Logistics ────────────────────────────────────────
    "ODFL", "SAIA", "JBHT", "CHRW", "LSTR",
    
    # ── Misc High-Volume NASDAQ ──────────────────────────────────────
    "SMCI", "SOUN", "MSTR", "CELH",
    "AXON", "CROX", "WING", "CASY", "TXRH",
    "MEDP", "LNTH", "MMSI", "NEOG",
    "WIX", "FIVN", "NICE",
]


class UniverseBuilder:
    """
    Dynamically select 180 volatile + 120 non-volatile NASDAQ stocks.
    
    Parameters
    ----------
    seed_pool : list[str]
        Starting pool of NASDAQ tickers to screen
    volatile_count : int
        Number of volatile stocks to select (default 180)
    nonvolatile_count : int
        Number of non-volatile stocks to select (default 120)
    atr_window : int
        ATR lookback period in days (default 20)
    min_adtv_usd : float
        Minimum average daily turnover in USD (default $1M)
    min_price : float
        Minimum stock price (default $5)
    min_history_days : int
        Minimum trading days of history required (default 60)
    """
    
    def __init__(
        self,
        seed_pool: Optional[list[str]] = None,
        volatile_count: int = 180,
        nonvolatile_count: int = 120,
        atr_window: int = 20,
        min_adtv_usd: float = 1_000_000,
        min_price: float = 5.0,
        min_history_days: int = 60,
    ):
        self.seed_pool = seed_pool or SEED_POOL
        self.volatile_count = volatile_count
        self.nonvolatile_count = nonvolatile_count
        self.atr_window = atr_window
        self.min_adtv = min_adtv_usd
        self.min_price = min_price
        self.min_history = min_history_days
    
    def select(
        self,
        daily_panels: dict[str, pd.DataFrame],
    ) -> tuple[list[str], list[str]]:
        """
        Select volatile and non-volatile stocks from daily OHLCV data.
        
        Parameters
        ----------
        daily_panels : dict
            Daily OHLCV panels: {"open": df, "high": df, ...}
            Each DataFrame: [date × ticker]
        
        Returns
        -------
        volatile : list[str] — Top N by ATR% (highest volatility)
        nonvolatile : list[str] — Bottom N by ATR% (lowest volatility)
        """
        close = daily_panels.get("close")
        high = daily_panels.get("high")
        low = daily_panels.get("low")
        volume = daily_panels.get("volume")
        
        if close is None or close.empty:
            log.error("No close data for universe screening")
            return [], []
        
        log.info("Universe screening: %d candidate tickers", len(close.columns))
        
        # ── Filter 1: Minimum history ────────────────────────────────────
        obs_count = close.notna().sum()
        has_history = obs_count[obs_count >= self.min_history].index.tolist()
        log.info("  After history filter (≥%d days): %d tickers",
                 self.min_history, len(has_history))
        
        # ── Filter 2: Minimum price ──────────────────────────────────────
        last_price = close.iloc[-1]
        above_price = last_price[last_price >= self.min_price].index.tolist()
        valid = list(set(has_history) & set(above_price))
        log.info("  After price filter (≥$%.0f): %d tickers",
                 self.min_price, len(valid))
        
        # ── Filter 3: Minimum ADTV ──────────────────────────────────────
        if volume is not None:
            dollar_vol = close * volume
            adtv = dollar_vol.rolling(20, min_periods=10).mean().iloc[-1]
            above_adtv = adtv[adtv >= self.min_adtv].index.tolist()
            valid = list(set(valid) & set(above_adtv))
            log.info("  After ADTV filter (≥$%.0fM): %d tickers",
                     self.min_adtv / 1_000_000, len(valid))
        
        if len(valid) < self.volatile_count + self.nonvolatile_count:
            log.warning(
                "  Only %d tickers pass filters (need %d). Relaxing criteria.",
                len(valid), self.volatile_count + self.nonvolatile_count
            )
        
        # ── Compute ATR% for ranking ─────────────────────────────────────
        atr_pct = self._compute_atr_pct(
            high[valid] if high is not None else None,
            low[valid] if low is not None else None,
            close[valid],
        )
        
        # Sort by ATR% descending (most volatile first)
        atr_ranked = atr_pct.sort_values(ascending=False)
        
        # ── Select top N volatile + bottom N non-volatile ────────────────
        n_vol = min(self.volatile_count, len(atr_ranked) // 2)
        n_nonvol = min(self.nonvolatile_count, len(atr_ranked) // 2)
        
        volatile = atr_ranked.head(n_vol).index.tolist()
        nonvolatile = atr_ranked.tail(n_nonvol).index.tolist()
        
        log.info("  Volatile %d (highest ATR%%): %s ... (ATR%% range: %.2f%% - %.2f%%)",
                 n_vol, volatile[:5], atr_ranked.iloc[0] * 100, atr_ranked.iloc[n_vol - 1] * 100)
        log.info("  Non-volatile %d (lowest ATR%%): %s ... (ATR%% range: %.2f%% - %.2f%%)",
                 n_nonvol, nonvolatile[:5], atr_ranked.iloc[-n_nonvol] * 100, atr_ranked.iloc[-1] * 100)
        
        return volatile, nonvolatile
    
    def _compute_atr_pct(
        self,
        high: Optional[pd.DataFrame],
        low: Optional[pd.DataFrame],
        close: pd.DataFrame,
    ) -> pd.Series:
        """
        Compute ATR as percentage of close (trailing average).
        
        ATR% = ATR(window) / Close
        Higher ATR% = more volatile stock.
        
        Returns pd.Series indexed by ticker, values = ATR%.
        """
        if high is not None and low is not None:
            # True Range = max(H-L, |H-Cprev|, |L-Cprev|)
            prev_close = close.shift(1)
            tr1 = high - low
            tr2 = (high - prev_close).abs()
            tr3 = (low - prev_close).abs()
            tr = pd.concat([tr1, tr2, tr3]).groupby(level=0).max()
        else:
            # Fallback: use absolute return as volatility proxy
            tr = close.pct_change(1).abs() * close
        
        atr = tr.rolling(self.atr_window, min_periods=10).mean()
        atr_pct = (atr / close.replace(0, np.nan)).iloc[-1]
        
        return atr_pct.dropna().sort_values(ascending=False)
    
    def get_full_universe(
        self,
        daily_panels: dict[str, pd.DataFrame],
    ) -> tuple[list[str], list[str], pd.Series]:
        """
        Get universe with ATR% scores for analysis.
        
        Returns
        -------
        volatile, nonvolatile, atr_scores
        """
        volatile, nonvolatile = self.select(daily_panels)
        
        close = daily_panels.get("close")
        high = daily_panels.get("high")
        low = daily_panels.get("low")
        
        valid = volatile + nonvolatile
        atr_scores = self._compute_atr_pct(
            high[valid] if high is not None else None,
            low[valid] if low is not None else None,
            close[valid],
        )
        
        return volatile, nonvolatile, atr_scores
