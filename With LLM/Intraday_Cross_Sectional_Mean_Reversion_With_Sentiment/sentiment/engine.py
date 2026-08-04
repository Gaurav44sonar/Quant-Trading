"""
sentiment/engine.py
===================
Decision engine that calculates confidence scores and decides trade confirmations.

Combines:
    - Existing signal (normal CDF of z-score)
    - Company-specific news sentiment (weighted by recency/decay)
    - Global market sentiment
    - Sector sentiment (mapped via SECTOR_MAP)

Features:
    - Configurable weights and thresholds
    - Time-decay: recent news has higher weight
    - Strong negative hard-rejections
    - Safe fallback if config or files are missing

Usage:
    engine = SentimentDecisionEngine(market="us")
    confirmed_picks, report_data = engine.evaluate(picks, target_date)
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone
import yaml
import numpy as np
import pandas as pd
from scipy.stats import norm

from alpha.stock_picker import SECTOR_MAP
from sentiment.collector import NewsCollector, Article
from sentiment.nlp import SentimentAnalyzer

log = logging.getLogger(__name__)


class SentimentDecisionEngine:
    """
    Evaluates and confirms stock picks using real-time news sentiment.
    """
    
    def __init__(self, market: str = "us", config_path: str = "config/config.yaml"):
        self.market = market.lower()
        self.config_path = config_path
        
        # Load config with fallback defaults
        self.config = self._load_config(config_path)
        
        # Initialize subcomponents
        cfg_filter = self.config.get("news_filter", {})
        sources = cfg_filter.get("sources", {})
        enabled_sources = []
        if sources.get("yahoo_rss", True):
            enabled_sources.append("yahoo_rss")
        if sources.get("google_rss", True):
            enabled_sources.append("google_rss")
        
        # Retrieve credentials from env if available for Alpaca News
        api_key = os.getenv("ALPACA_API_KEY", "")
        secret_key = os.getenv("ALPACA_SECRET_KEY", "")
        if sources.get("alpaca_news", True) and api_key and secret_key:
            enabled_sources.append("alpaca")
            
        self.max_age_hours = cfg_filter.get("max_age_hours", 24)
        self.decay_halflife = cfg_filter.get("decay_halflife_hours", 6)
        
        self.collector = NewsCollector(
            sources=enabled_sources,
            max_age_hours=self.max_age_hours,
            alpaca_api_key=api_key,
            alpaca_secret_key=secret_key,
        )
        
        self.analyzer = SentimentAnalyzer(use_vader=True)
        
        # Decision engine weights & thresholds
        self.enabled = self.config.get("enabled", True)
        self.min_confidence = self.config.get("min_confidence_threshold", 0.5)
        self.strong_neg_threshold = self.config.get("strong_negative_threshold", -0.5)
        
        weights = self.config.get("weights", {})
        self.w_signal = weights.get("signal", 0.40)
        self.w_news = weights.get("news", 0.30)
        self.w_market = weights.get("market", 0.15)
        self.w_sector = weights.get("sector", 0.15)
        
        # Normalize weights to sum to 1.0
        total_w = self.w_signal + self.w_news + self.w_market + self.w_sector
        if total_w > 0:
            self.w_signal /= total_w
            self.w_news /= total_w
            self.w_market /= total_w
            self.w_sector /= total_w
            
    def _load_config(self, path: str) -> dict:
        """Load YAML config and extract sentiment configuration block."""
        default_config = {
            "enabled": True,
            "min_confidence_threshold": 0.5,
            "strong_negative_threshold": -0.5,
            "weights": {
                "signal": 0.40,
                "news": 0.30,
                "market": 0.15,
                "sector": 0.15,
            },
            "news_filter": {
                "max_age_hours": 24,
                "decay_halflife_hours": 6,
                "sources": {
                    "yahoo_rss": True,
                    "google_rss": True,
                    "alpaca_news": True,
                }
            }
        }
        
        if not os.path.exists(path):
            log.debug("[SENTIMENT] Config file not found at %s. Using defaults.", path)
            return default_config
            
        try:
            with open(path, "r") as f:
                full_config = yaml.safe_load(f) or {}
                sentiment_block = full_config.get("sentiment", {})
                
                # Merge defaults
                merged = default_config.copy()
                for k, v in sentiment_block.items():
                    if isinstance(v, dict) and k in merged:
                        merged[k].update(v)
                    else:
                        merged[k] = v
                return merged
        except Exception as e:
            log.warning("[SENTIMENT] Failed to parse config file: %s. Using defaults.", e)
            return default_config

    def evaluate(self, picks: list[dict], target_date: datetime) -> tuple[list[dict], dict]:
        """
        Evaluate candidate picks using news sentiment.
        
        Returns:
            confirmed_picks: filtered list of picks containing only confirmed trades
            report_data: a dictionary containing stats and detailed logs for the report
        """
        if not self.enabled:
            log.info("[SENTIMENT] Sentiment module disabled. Confirming all picks.")
            return picks, {}
            
        tickers = [p["ticker"] for p in picks]
        if not tickers:
            return [], {}
            
        log.info("[SENTIMENT] Evaluating confirmation for %d candidates...", len(tickers))
        
        # 1. Fetch all relevant articles
        articles = self.collector.fetch(tickers=tickers, market_news=True, market=self.market)
        
        # 2. Analyze sentiment of each article
        ticker_articles: dict[str, list[tuple[Article, float, float]]] = {t: [] for t in tickers}
        market_articles: list[tuple[Article, float, float]] = []
        
        for art in articles:
            res = self.analyzer.analyze(art.headline + " " + art.summary)
            # Apply time decay weight
            age = art.age_hours
            weight = 2 ** (-age / self.decay_halflife) if age <= self.max_age_hours else 0.0
            
            if art.ticker:
                if art.ticker in ticker_articles:
                    ticker_articles[art.ticker].append((art, res.polarity, weight))
            else:
                market_articles.append((art, res.polarity, weight))
                
        # 3. Aggregate company sentiments
        company_sentiments: dict[str, float] = {}
        company_counts: dict[str, int] = {}
        for ticker in tickers:
            arts_data = ticker_articles[ticker]
            company_counts[ticker] = len(arts_data)
            if arts_data:
                weighted_sum = sum(pol * w for _, pol, w in arts_data)
                weight_sum = sum(w for _, _, w in arts_data)
                company_sentiments[ticker] = (weighted_sum / weight_sum) if weight_sum > 0 else 0.0
            else:
                company_sentiments[ticker] = 0.0
                
        # 4. Aggregate market-wide sentiment
        if market_articles:
            weighted_sum = sum(pol * w for _, pol, w in market_articles)
            weight_sum = sum(w for _, _, w in market_articles)
            market_sentiment = (weighted_sum / weight_sum) if weight_sum > 0 else 0.0
        else:
            market_sentiment = 0.0
            
        # 5. Aggregate sector sentiment
        sector_sentiments: dict[str, list[float]] = {}
        for ticker, pol in company_sentiments.items():
            sector = SECTOR_MAP.get(ticker, "Other")
            if sector not in sector_sentiments:
                sector_sentiments[sector] = []
            if company_counts[ticker] > 0:  # Only count if the stock actually had news
                sector_sentiments[sector].append(pol)
                
        sector_avg: dict[str, float] = {}
        for sector, pols in sector_sentiments.items():
            sector_avg[sector] = float(np.mean(pols)) if pols else 0.0
            
        # 6. Evaluate picks and calculate scores
        confirmed_picks = []
        stock_news_details = []
        stats = {
            "confirmed": 0,
            "rejected": 0,
            "held": 0,
            "total_articles": len(articles),
            "dup_removed": self.collector.stats["duplicates_removed"],
            "avg_sentiment": float(np.mean(list(company_sentiments.values()))) if company_sentiments else 0.0,
            "market_sentiment": market_sentiment,
            "sector_sentiment": sector_avg,
        }
        
        for pick in picks:
            ticker = pick["ticker"]
            z_score = pick.get("score", 0.0)
            
            # Map signal z-score to probability [0, 1] using standard normal CDF
            signal_prob = float(norm.cdf(z_score))
            
            # Retrieve company, sector, and market sentiment scores and scale from [-1, 1] to [0, 1]
            comp_pol = company_sentiments[ticker]
            comp_score = (comp_pol + 1) / 2
            
            mkt_score = (market_sentiment + 1) / 2
            
            sector = SECTOR_MAP.get(ticker, "Other")
            sect_pol = sector_avg.get(sector, 0.0)
            sect_score = (sect_pol + 1) / 2
            
            # Calculate final weighted confidence score
            confidence = (
                self.w_signal * signal_prob +
                self.w_news * comp_score +
                self.w_market * mkt_score +
                self.w_sector * sect_score
            )
            
            # Apply decision rules
            reason = ""
            if comp_pol <= self.strong_neg_threshold:
                decision = "REJECT"
                reason = f"Strong negative news sentiment ({comp_pol:.2f})"
                stats["rejected"] += 1
            elif confidence >= self.min_confidence:
                decision = "CONFIRM"
                reason = f"Confidence score ({confidence:.2f}) meets threshold"
                stats["confirmed"] += 1
            else:
                decision = "HOLD"
                reason = f"Borderline confidence score ({confidence:.2f})"
                stats["held"] += 1
                
            # Log the detailed decision
            log.info("  %s: Signal=%+.2fz -> Conf=%.2f | Sentiment: Ticker=%+.2f, Mkt=%+.2f, Sect=%+.2f | Decision: %s (%s)",
                     ticker, z_score, confidence, comp_pol, market_sentiment, sect_pol, decision, reason)
            
            # Store detail
            detail = {
                "ticker": ticker,
                "z_score": z_score,
                "company_sentiment": comp_pol,
                "sector": sector,
                "sector_sentiment": sect_pol,
                "market_sentiment": market_sentiment,
                "confidence": confidence,
                "decision": decision,
                "reason": reason,
                "articles_count": company_counts[ticker],
            }
            stock_news_details.append(detail)
            
            if decision == "CONFIRM":
                # Add sentiment metadata to pick dict for tracking
                confirmed_pick = pick.copy()
                confirmed_pick["sentiment_score"] = comp_pol
                confirmed_pick["sentiment_confidence"] = confidence
                # Add the most relevant headline for LLM context
                if ticker_articles[ticker]:
                    best_art = max(ticker_articles[ticker], key=lambda x: x[2])  # highest weight = most recent
                    confirmed_pick["news_headline"] = best_art[0].headline
                confirmed_picks.append(confirmed_pick)
                
        # Package report summary
        report_summary = {
            "stats": stats,
            "stock_details": stock_news_details,
            "sources_used": self.collector.stats["sources_used"],
        }
        
        log.info("[SENTIMENT] Confirmation complete. Confirmed: %d, Rejected: %d, Held: %d",
                 stats["confirmed"], stats["rejected"], stats["held"])
        
        return confirmed_picks, report_summary
