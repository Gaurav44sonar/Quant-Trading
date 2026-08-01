"""
sentiment/nlp.py
================
Tiered NLP Sentiment Analyzer for financial news headlines.

Architecture:
    Tier 1 (Default, zero-dep): Financial sentiment lexicon (~200 scored words/phrases)
        + regex-based critical event detection (earnings, bankruptcy, FDA, etc.)
    Tier 2 (Optional): VADER sentiment (auto-detected, handles negation/modifiers)

The analyzer returns per-article:
    - polarity: float [-1.0, +1.0]
    - confidence: float [0.0, 1.0]
    - detected_events: list[str]

Usage:
    analyzer = SentimentAnalyzer()
    result = analyzer.analyze("NVDA beats earnings expectations, raises guidance")
    # result = {"polarity": 0.85, "confidence": 0.92, "events": ["earnings_beat", "guidance_raise"]}
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Financial Sentiment Lexicon
# ─────────────────────────────────────────────────────────────────────────────

# Scored phrases (checked first, before single words)
# Format: (pattern, score) where score ∈ [-1, 1]
PHRASE_LEXICON: list[tuple[str, float]] = [
    # Strong positive
    (r"\bbeats?\s+(?:earnings|estimates|expectations|consensus)\b", +0.9),
    (r"\b(?:raises?|lifts?|hikes?)\s+(?:guidance|outlook|forecast|dividend)\b", +0.85),
    (r"\b(?:record|all[- ]?time)\s+(?:high|revenue|profit|earnings)\b", +0.8),
    (r"\bupgrade[ds]?\s+(?:to|from)\s+(?:buy|outperform|overweight)\b", +0.8),
    (r"\bstrong(?:er)?\s+(?:than\s+expected|results|earnings|growth)\b", +0.75),
    (r"\b(?:fda|sec)\s+(?:approv|clear)\w*\b", +0.85),
    (r"\bprice\s+target\s+(?:raised?|increased?|hiked?)\b", +0.7),
    (r"\bstock\s+(?:buyback|repurchase)\b", +0.6),
    (r"\bbetter[- ]than[- ]expected\b", +0.75),
    (r"\bblowout\s+(?:quarter|earnings|results)\b", +0.85),
    (r"\bnarrows?\s+(?:loss|losses)\b", +0.4),
    
    # Strong negative
    (r"\bmisses?\s+(?:earnings|estimates|expectations|consensus)\b", -0.85),
    (r"\b(?:lowers?|cuts?|slashes?)\s+(?:guidance|outlook|forecast|dividend)\b", -0.85),
    (r"\bdowngrade[ds]?\s+(?:to|from)\s+(?:sell|underperform|underweight)\b", -0.8),
    (r"\b(?:files?\s+for|declares?)\s+bankruptcy\b", -0.95),
    (r"\bchapter\s+(?:7|11|15)\b", -0.9),
    (r"\b(?:sec|doj|fbi)\s+(?:investigat|prob|charg|su)\w*\b", -0.85),
    (r"\bclass[- ]?action\s+(?:lawsuit|suit)\b", -0.8),
    (r"\bfraud(?:ulent)?\s+(?:charges?|allegations?|scheme)\b", -0.9),
    (r"\bdata\s+breach\b", -0.7),
    (r"\b(?:massive|significant|major)\s+layoffs?\b", -0.75),
    (r"\bprice\s+target\s+(?:lowered?|decreased?|cut|slashed?)\b", -0.7),
    (r"\bworse[- ]than[- ]expected\b", -0.75),
    (r"\bgoing\s+concern\b", -0.85),
    (r"\bdelisted?\b", -0.9),
    (r"\brecall(?:s|ed|ing)?\b", -0.6),
    (r"\bwidening\s+(?:loss|losses)\b", -0.7),
    
    # Moderate
    (r"\bacquir(?:es?|ed|ing|ition)\b", +0.3),
    (r"\bmerger\s+(?:with|agreement|deal)\b", +0.3),
    (r"\bpartnership\s+with\b", +0.35),
    (r"\bnew\s+(?:product|launch|contract)\b", +0.4),
    (r"\bexpand(?:s|ed|ing)?\s+(?:into|operations|capacity)\b", +0.35),
    (r"\b(?:rate|rates?)\s+(?:cut|hike|hold|unchanged)\b", 0.0),  # Neutral — context dependent
]

# Single-word lexicon
# Format: {word: score}
WORD_LEXICON: dict[str, float] = {
    # ── Positive ──
    "beat": +0.7, "beats": +0.7, "exceeded": +0.65, "exceeds": +0.65,
    "surge": +0.6, "surges": +0.6, "surging": +0.6, "soar": +0.65, "soars": +0.65, "soaring": +0.65,
    "rally": +0.55, "rallies": +0.55, "rallying": +0.55,
    "upgrade": +0.7, "upgraded": +0.7, "upgrades": +0.7,
    "bullish": +0.6, "optimistic": +0.5, "outperform": +0.65, "overweight": +0.55,
    "profit": +0.4, "profitable": +0.45, "profitability": +0.45,
    "growth": +0.35, "growing": +0.35, "grew": +0.35,
    "positive": +0.4, "gains": +0.4, "gain": +0.4, "gaining": +0.4,
    "recovery": +0.4, "recovering": +0.4, "recovered": +0.4,
    "breakthrough": +0.6, "innovation": +0.4, "innovative": +0.4,
    "strong": +0.35, "strength": +0.35, "robust": +0.4,
    "approval": +0.55, "approved": +0.55, "approves": +0.55,
    "dividend": +0.3, "buyback": +0.35, "repurchase": +0.35,
    "outpaced": +0.45, "outperformed": +0.5,
    "boom": +0.5, "booming": +0.5, "blockbuster": +0.6,
    "record": +0.4, "milestone": +0.4,
    "win": +0.4, "wins": +0.4, "winning": +0.4, "won": +0.4,
    
    # ── Negative ──
    "miss": -0.65, "missed": -0.65, "misses": -0.65, "missing": -0.3,
    "decline": -0.5, "declines": -0.5, "declining": -0.5, "declined": -0.5,
    "plunge": -0.7, "plunges": -0.7, "plunging": -0.7, "plunged": -0.7,
    "crash": -0.75, "crashes": -0.75, "crashing": -0.75, "crashed": -0.75,
    "downgrade": -0.7, "downgraded": -0.7, "downgrades": -0.7,
    "bearish": -0.6, "pessimistic": -0.5, "underperform": -0.6, "underweight": -0.55,
    "loss": -0.45, "losses": -0.45, "losing": -0.4,
    "bankruptcy": -0.9, "bankrupt": -0.9, "insolvent": -0.85,
    "investigation": -0.65, "investigated": -0.65, "probe": -0.6,
    "lawsuit": -0.6, "sued": -0.6, "suing": -0.6, "litigation": -0.55,
    "fraud": -0.85, "fraudulent": -0.85, "scam": -0.8,
    "layoff": -0.55, "layoffs": -0.55, "restructuring": -0.35,
    "recall": -0.5, "recalled": -0.5,
    "debt": -0.3, "default": -0.7, "defaults": -0.7, "defaulted": -0.7,
    "warning": -0.45, "warns": -0.5, "warned": -0.5,
    "risk": -0.25, "risky": -0.3, "risks": -0.25,
    "weak": -0.4, "weakness": -0.4, "weakening": -0.4,
    "sell": -0.35, "selling": -0.3, "selloff": -0.55,
    "cut": -0.35, "cuts": -0.35, "slashed": -0.5,
    "suspension": -0.6, "suspended": -0.6, "halt": -0.5, "halted": -0.5,
    "tumble": -0.6, "tumbles": -0.6, "tumbling": -0.6,
    "drop": -0.4, "drops": -0.4, "dropping": -0.4, "dropped": -0.4,
    "fall": -0.35, "falls": -0.35, "falling": -0.35, "fell": -0.35,
    "sink": -0.55, "sinks": -0.55, "sinking": -0.55,
    "concern": -0.3, "concerns": -0.3, "worried": -0.35, "worry": -0.3,
    "volatile": -0.15, "volatility": -0.15, "uncertainty": -0.25,
    "disappointing": -0.55, "disappointed": -0.55, "disappoints": -0.55,
    "negative": -0.4, "worst": -0.5,
    "delayed": -0.35, "delay": -0.35, "delays": -0.35,
    "shortage": -0.4, "shortages": -0.4,
    
    # ── Neutral/Contextual ──
    "steady": +0.1, "stable": +0.1, "unchanged": 0.0,
    "mixed": 0.0, "flat": 0.0,
}

# Negation words that flip sentiment
NEGATION_WORDS = {
    "not", "no", "never", "neither", "nor", "none", "nobody",
    "nothing", "nowhere", "hardly", "barely", "scarcely",
    "doesn't", "didn't", "don't", "won't", "wouldn't", "couldn't",
    "shouldn't", "can't", "cannot", "isn't", "aren't", "wasn't", "weren't",
    "despite", "fails", "failed", "fail",
}

# Intensifier words that amplify sentiment
INTENSIFIERS = {
    "very": 1.3, "extremely": 1.5, "significantly": 1.4,
    "substantially": 1.3, "massively": 1.5, "sharply": 1.4,
    "dramatically": 1.5, "hugely": 1.4, "major": 1.3,
    "big": 1.2, "huge": 1.3, "massive": 1.4,
}


# ─────────────────────────────────────────────────────────────────────────────
# Event Detection Patterns
# ─────────────────────────────────────────────────────────────────────────────

EVENT_PATTERNS: list[tuple[str, str, float]] = [
    # (regex_pattern, event_name, sentiment_override)
    (r"\b(?:beats?|topped?|exceeded?)\s+(?:earnings|estimates|expectations|EPS)\b", "earnings_beat", +0.85),
    (r"\b(?:miss(?:es|ed)?|fell\s+short)\s+(?:earnings|estimates|expectations|EPS)\b", "earnings_miss", -0.85),
    (r"\b(?:raises?|lifts?|hikes?|increases?)\s+(?:guidance|outlook|forecast)\b", "guidance_raise", +0.8),
    (r"\b(?:lowers?|cuts?|slashes?|reduces?)\s+(?:guidance|outlook|forecast)\b", "guidance_lower", -0.8),
    (r"\b(?:files?\s+for\s+)?bankruptcy\b", "bankruptcy", -0.95),
    (r"\bchapter\s+(?:7|11|15)\b", "bankruptcy_filing", -0.9),
    (r"\b(?:acquir|acquisition|merger|takeover|buyout)\w*\b", "m_and_a", +0.3),
    (r"\b(?:fda|sec)\s+approv\w*\b", "regulatory_approval", +0.8),
    (r"\b(?:sec|doj|fbi|ftc)\s+(?:investigat|prob|charg)\w*\b", "regulatory_investigation", -0.8),
    (r"\bclass[- ]?action\b", "class_action_lawsuit", -0.75),
    (r"\blayoffs?\b", "layoffs", -0.5),
    (r"\bstock\s+split\b", "stock_split", +0.3),
    (r"\b(?:dividend|special\s+dividend)\b", "dividend", +0.3),
    (r"\b(?:ceo|cfo|cto|coo)\s+(?:resign|step\w*\s+down|depart|fired|terminat)\w*\b", "executive_departure", -0.5),
    (r"\b(?:product|drug|device)\s+(?:launch|release|approv)\w*\b", "product_launch", +0.5),
    (r"\b(?:analyst|price\s+target)\s+(?:upgrade|raise)\w*\b", "analyst_upgrade", +0.7),
    (r"\b(?:analyst|price\s+target)\s+(?:downgrade|cut|lower)\w*\b", "analyst_downgrade", -0.7),
    (r"\binsider\s+(?:buy|purchas)\w*\b", "insider_buying", +0.5),
    (r"\binsider\s+(?:sell|sold|dump)\w*\b", "insider_selling", -0.4),
    (r"\bshort\s+(?:sell|interest|squeeze)\w*\b", "short_interest", -0.2),
    (r"\b(?:tariff|trade\s+war|sanctions?)\b", "trade_policy", -0.4),
    (r"\b(?:rate\s+cut|rate\s+hike|interest\s+rate)\b", "interest_rate", 0.0),
]


# ─────────────────────────────────────────────────────────────────────────────
# Sentiment Result
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SentimentResult:
    """Result of sentiment analysis for a single article."""
    polarity: float = 0.0       # [-1.0, +1.0]
    confidence: float = 0.0     # [0.0, 1.0]
    events: list[str] = field(default_factory=list)
    method: str = "lexicon"     # "lexicon", "vader", or "combined"


# ─────────────────────────────────────────────────────────────────────────────
# Sentiment Analyzer
# ─────────────────────────────────────────────────────────────────────────────

class SentimentAnalyzer:
    """
    Tiered financial news sentiment analyzer.
    
    Tier 1: Financial lexicon (always available)
    Tier 2: VADER sentiment (optional, auto-detected)
    
    Combines both tiers when available, with lexicon handling
    domain-specific terms and VADER handling general language nuances.
    """
    
    def __init__(self, use_vader: bool = True):
        self._vader = None
        if use_vader:
            try:
                from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
                self._vader = SentimentIntensityAnalyzer()
                log.info("[SENTIMENT] VADER sentiment analyzer loaded (Tier 2)")
            except ImportError:
                log.info("[SENTIMENT] VADER not installed, using lexicon only (Tier 1)")
        
        self._phrase_patterns = [
            (re.compile(pattern, re.IGNORECASE), score)
            for pattern, score in PHRASE_LEXICON
        ]
        
        self._event_patterns = [
            (re.compile(pattern, re.IGNORECASE), name, score)
            for pattern, name, score in EVENT_PATTERNS
        ]
    
    def analyze(self, text: str) -> SentimentResult:
        """
        Analyze sentiment of a single text (headline + summary).
        
        Returns SentimentResult with polarity, confidence, and detected events.
        """
        if not text or not text.strip():
            return SentimentResult()
        
        text_lower = text.lower().strip()
        
        # 1. Detect critical events first (these override general sentiment)
        events = []
        event_sentiment = 0.0
        event_count = 0
        for pattern, name, score in self._event_patterns:
            if pattern.search(text_lower):
                events.append(name)
                event_sentiment += score
                event_count += 1
        
        # 2. Lexicon-based sentiment
        lexicon_score = self._lexicon_score(text_lower)
        
        # 3. VADER sentiment (if available)
        vader_score = 0.0
        has_vader = False
        if self._vader:
            try:
                vs = self._vader.polarity_scores(text)
                vader_score = vs["compound"]  # Already in [-1, 1]
                has_vader = True
            except Exception:
                pass
        
        # 4. Combine scores
        if event_count > 0:
            # Events detected → event sentiment dominates
            event_avg = event_sentiment / event_count
            if has_vader:
                polarity = 0.5 * event_avg + 0.25 * lexicon_score + 0.25 * vader_score
                method = "combined+events"
            else:
                polarity = 0.6 * event_avg + 0.4 * lexicon_score
                method = "lexicon+events"
        elif has_vader:
            # No events, combine lexicon + VADER
            polarity = 0.45 * lexicon_score + 0.55 * vader_score
            method = "combined"
        else:
            # Lexicon only
            polarity = lexicon_score
            method = "lexicon"
        
        # Clamp to [-1, 1]
        polarity = max(-1.0, min(1.0, polarity))
        
        # Confidence: based on signal strength and agreement
        confidence = self._compute_confidence(polarity, lexicon_score, vader_score, has_vader, event_count)
        
        return SentimentResult(
            polarity=polarity,
            confidence=confidence,
            events=events,
            method=method,
        )
    
    def _lexicon_score(self, text: str) -> float:
        """Score text using financial lexicon (phrases + words)."""
        score = 0.0
        matches = 0
        
        # 1. Check phrase patterns first (higher priority)
        for pattern, phrase_score in self._phrase_patterns:
            if pattern.search(text):
                score += phrase_score
                matches += 1
        
        # 2. Word-level scoring with negation and intensifier handling
        words = re.findall(r"\b[a-z]+(?:'[a-z]+)?\b", text)
        
        negation_active = False
        intensifier = 1.0
        
        for i, word in enumerate(words):
            # Check for negation
            if word in NEGATION_WORDS:
                negation_active = True
                continue
            
            # Check for intensifiers
            if word in INTENSIFIERS:
                intensifier = INTENSIFIERS[word]
                continue
            
            # Score the word
            if word in WORD_LEXICON:
                word_score = WORD_LEXICON[word] * intensifier
                if negation_active:
                    word_score *= -0.75  # Partial flip
                score += word_score
                matches += 1
            
            # Reset modifiers after use
            negation_active = False
            intensifier = 1.0
        
        # Normalize: average score, clamped to [-1, 1]
        if matches > 0:
            normalized = score / (matches ** 0.5)  # Sqrt normalization prevents runaway
            return max(-1.0, min(1.0, normalized))
        
        return 0.0
    
    def _compute_confidence(
        self,
        polarity: float,
        lexicon_score: float,
        vader_score: float,
        has_vader: bool,
        event_count: int,
    ) -> float:
        """Compute confidence score [0, 1] based on signal strength and agreement."""
        # Base confidence from signal strength
        strength = abs(polarity)
        
        # Agreement bonus if both methods agree
        agreement_bonus = 0.0
        if has_vader:
            if (lexicon_score > 0 and vader_score > 0) or (lexicon_score < 0 and vader_score < 0):
                agreement_bonus = 0.15
            elif (lexicon_score > 0 and vader_score < 0) or (lexicon_score < 0 and vader_score > 0):
                agreement_bonus = -0.1  # Disagreement penalty
        
        # Event bonus (detected events = higher confidence)
        event_bonus = min(0.2, event_count * 0.1)
        
        confidence = strength + agreement_bonus + event_bonus
        return max(0.0, min(1.0, confidence))
    
    def analyze_batch(self, texts: list[str]) -> list[SentimentResult]:
        """Analyze multiple texts."""
        return [self.analyze(t) for t in texts]
