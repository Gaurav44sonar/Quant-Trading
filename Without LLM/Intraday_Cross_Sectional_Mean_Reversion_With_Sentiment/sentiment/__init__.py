"""
sentiment/__init__.py
=====================
Real-Time News Sentiment Analysis Module.

Provides a modular, fault-tolerant sentiment confirmation layer
for the intraday mean-reversion trading pipeline.

Components:
    - NewsCollector: Fetches news from multiple sources (Yahoo RSS, Google RSS, Alpaca)
    - SentimentAnalyzer: NLP processing (VADER + Financial Lexicon + Event Detection)
    - SentimentDecisionEngine: Aggregation, time decay, confidence scoring, trade confirmation
"""

from sentiment.collector import NewsCollector, Article
from sentiment.nlp import SentimentAnalyzer
from sentiment.engine import SentimentDecisionEngine

__all__ = [
    "NewsCollector",
    "SentimentAnalyzer", 
    "SentimentDecisionEngine",
    "Article",
]
