"""
nse_pipeline/universe_nse.py
=============================
Seed pool of ~200 liquid NSE-listed stocks for intraday mean reversion.

Covers NIFTY 50, NIFTY Next 50, popular F&O stocks, and high-volume
mid-caps. Final selection of 50 volatile + 50 non-volatile is done
dynamically by ATR% ranking (same logic as NASDAQ universe builder).

Usage
-----
    from nse_pipeline.universe_nse import NSE_SEED_POOL
"""

# ─────────────────────────────────────────────────────────────────────────────
# ~200 liquid NSE-listed stocks (use WITHOUT .NS suffix — suffix added by fetcher)
# ─────────────────────────────────────────────────────────────────────────────

NSE_SEED_POOL = [
    # ══════════════════════════════════════════════════════════════════════
    # NIFTY 50 — India's Blue Chips
    # ══════════════════════════════════════════════════════════════════════

    # ── Banking / Financials ────────────────────────────────────────────
    "HDFCBANK", "ICICIBANK", "SBIN", "KOTAKBANK", "AXISBANK",
    "INDUSINDBK", "BANDHANBNK", "FEDERALBNK", "IDFCFIRSTB", "PNB",
    "BANKBARODA", "CANBK", "AUBANK", "RBLBANK",

    # ── IT / Software ──────────────────────────────────────────────────
    "TCS", "INFY", "WIPRO", "HCLTECH", "TECHM",
    "LTIM", "MPHASIS", "COFORGE", "PERSISTENT", "LTTS",

    # ── Reliance / Conglomerates ───────────────────────────────────────
    "RELIANCE", "ITC", "HINDUNILVR", "BAJFINANCE", "BAJFINSV",

    # ── Auto ──────────────────────────────────────────────────────────
    "TATAMOTORS", "M&M", "MARUTI", "BAJAJ-AUTO", "HEROMOTOCO",
    "EICHERMOT", "ASHOKLEY", "TVSMOTOR", "BALKRISIND",

    # ── Pharma / Healthcare ──────────────────────────────────────────
    "SUNPHARMA", "DRREDDY", "CIPLA", "DIVISLAB", "APOLLOHOSP",
    "BIOCON", "LUPIN", "AUROPHARMA", "IPCALAB", "LALPATHLAB",
    "TORNTPHARM", "NATCOPHARMA", "GRANULES", "ALKEM",

    # ── Metals / Mining ──────────────────────────────────────────────
    "TATASTEEL", "HINDALCO", "JSWSTEEL", "VEDL", "COALINDIA",
    "NMDC", "NATIONALUM", "SAIL", "HINDCOPPER",

    # ── Oil / Gas / Energy ──────────────────────────────────────────
    "ONGC", "BPCL", "IOC", "GAIL", "NTPC",
    "POWERGRID", "ADANIGREEN", "TATAPOWER", "NHPC", "IREDA",

    # ── Cement / Infrastructure ─────────────────────────────────────
    "ULTRACEMCO", "SHREECEM", "AMBUJACEM", "ACC",
    "ADANIENT", "ADANIPORTS", "LT", "GRASIM",

    # ── FMCG / Consumer ────────────────────────────────────────────
    "NESTLEIND", "BRITANNIA", "DABUR", "MARICO", "GODREJCP",
    "COLPAL", "TATACONSUM", "VBL", "UNITDSPR",

    # ── Telecom ────────────────────────────────────────────────────
    "BHARTIARTL", "IDEA",

    # ══════════════════════════════════════════════════════════════════════
    # NIFTY NEXT 50 / F&O POPULAR — Mid & Large Caps
    # ══════════════════════════════════════════════════════════════════════

    # ── Insurance ──────────────────────────────────────────────────
    "SBILIFE", "HDFCLIFE", "ICICIPRULI",

    # ── NBFC / Financial Services ──────────────────────────────────
    "CHOLAFIN", "SHRIRAMFIN", "MUTHOOTFIN", "MANAPPURAM",
    "LICHSGFIN", "PEL", "RECLTD", "PFC", "IRFC",

    # ── Chemicals / Speciality ─────────────────────────────────────
    "PIDILITIND", "SRF", "AARTIIND", "DEEPAKNTR",
    "CLEAN", "ATUL", "NAVINFLUOR",

    # ── Industrials / Capital Goods ────────────────────────────────
    "SIEMENS", "ABB", "HAVELLS", "BHEL", "BEL",
    "HAL", "CUMMINSIND", "VOLTAS",

    # ── Real Estate ───────────────────────────────────────────────
    "DLF", "OBEROIRLTY", "GODREJPROP", "PRESTIGE", "PHOENIXLTD",

    # ── Media / Entertainment ─────────────────────────────────────
    "ZEEL", "PVR",

    # ── Retail / E-Commerce ───────────────────────────────────────
    "DMART", "TRENT", "NYKAA",

    # ── Travel / Hotels ───────────────────────────────────────────
    "INDHOTEL", "IRCTC", "MAKEMYTRIP",

    # ══════════════════════════════════════════════════════════════════════
    # HIGH-VOLUME MID-CAPS & SPECULATIVE — Intraday Favorites
    # ══════════════════════════════════════════════════════════════════════

    # ── PSU Banks (high intraday volatility) ──────────────────────
    "UNIONBANK", "IOB", "CENTRALBK", "INDIANB",
    "MAHABANK", "UCOBANK", "BANKINDIA",

    # ── Defence / Railway (thematic) ─────────────────────────────
    "MAZAGON", "COCHINSHIP", "GRSE", "RVNL", "IRCON",

    # ── Sugar / Ethanol ──────────────────────────────────────────
    "BALRAMCHIN", "TRIVENI", "RENUKA",

    # ── Power / Renewable ────────────────────────────────────────
    "ADANIPOWER", "JPPOWER", "SUZLON", "TORNTPOWER",

    # ── Misc High Volume ─────────────────────────────────────────
    "ZOMATO", "PAYTM", "POLICYBZR", "JIOFIN",
    "IDEA", "YESBANK", "GMRAIRPORT",
    "DIXON", "KAYNES", "CDSL",
]
