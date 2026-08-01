"""
ai_decision/schemas.py
======================
Pydantic data schemas for LLM decision layer inputs and outputs.
"""

from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field


# ── Entry Schemas ─────────────────────────────────────────────────────────────

class AlphaSignals(BaseModel):
    composite_score: float = Field(..., description="Composite alpha z-score")
    P1_overnight_gap: Optional[float] = None
    P2_prev_day_momentum: Optional[float] = None
    P3_volume_surge: Optional[float] = None
    P4_relative_strength: Optional[float] = None
    P5_range_expansion: Optional[float] = None
    P6_close_location: Optional[float] = None
    C1_opening_bar_reversal: Optional[float] = None
    C2_opening_volume: Optional[float] = None
    C3_gap_fill_speed: Optional[float] = None


class PickItem(BaseModel):
    symbol: str
    entry_price: float
    proposed_shares: int
    group: str = "volatile"
    atr_value: Optional[float] = None
    alpha_signals: AlphaSignals


class MarketContext(BaseModel):
    index_ticker: str = "QQQ"
    index_price: Optional[float] = None
    index_intraday_return_pct: Optional[float] = None
    market_tz: str = "US/Eastern"


class PortfolioContext(BaseModel):
    capital: float
    existing_positions: int = 0
    current_exposure_pct: float = 0.0


class EntryRequest(BaseModel):
    task: str = "entry_validation"
    timestamp: str
    market: MarketContext
    portfolio: PortfolioContext
    picks: List[PickItem]


class EntryDecisionItem(BaseModel):
    symbol: str
    action: str = Field(..., description="BUY, HOLD, or REDUCE")
    confidence: int = Field(..., ge=0, le=100, description="0-100 score")
    position_multiplier: float = Field(1.0, ge=0.0, le=2.0, description="0.0 to 2.0 multiplier on position size")
    reasoning: List[str] = Field(default_factory=list)


class EntryResponse(BaseModel):
    decisions: List[EntryDecisionItem]


# ── Exit Schemas ──────────────────────────────────────────────────────────────

class PositionStateItem(BaseModel):
    symbol: str
    entry_price: float
    current_price: float
    bar_high: float
    bar_low: float
    qty: int
    pnl_pct: float
    high_water: float
    minutes_held: float
    sl1_done: bool = False
    pt1_done: bool = False
    atr_value: Optional[float] = None


class ExitRequest(BaseModel):
    task: str = "exit_validation"
    timestamp: str
    market: MarketContext
    portfolio: PortfolioContext
    positions: List[PositionStateItem]


class ExitDecisionItem(BaseModel):
    symbol: str
    action: str = Field(..., description="HOLD, SELL, REDUCE, or TIGHTEN_STOPS")
    confidence: int = Field(..., ge=0, le=100)
    adjusted_trail_trigger: Optional[float] = Field(None, description="Optional adjusted trailing stop trigger, e.g. 0.01")
    reasoning: List[str] = Field(default_factory=list)


class ExitResponse(BaseModel):
    exit_decisions: List[ExitDecisionItem]
