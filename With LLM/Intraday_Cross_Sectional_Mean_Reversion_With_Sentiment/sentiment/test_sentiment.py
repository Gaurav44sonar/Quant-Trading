"""
sentiment/test_sentiment.py
===========================
Unit tests for the news sentiment analysis package.

Tests:
    - Article dataclass and age helper
    - SentimentAnalyzer lexicon scoring and event detection
    - SentimentDecisionEngine aggregation, normalization, and decision rules
"""

import unittest
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch

from sentiment.collector import Article
from sentiment.nlp import SentimentAnalyzer, SentimentResult
from sentiment.engine import SentimentDecisionEngine


class TestArticle(unittest.TestCase):
    def test_age_hours(self):
        now = datetime.now(timezone.utc)
        art = Article(
            headline="Test",
            published=now - timedelta(hours=2.5),
            url="http://example.com"
        )
        self.assertAlmostEqual(art.age_hours, 2.5, places=2)


class TestSentimentAnalyzer(unittest.TestCase):
    def setUp(self):
        self.analyzer = SentimentAnalyzer(use_vader=False)

    def test_positive_headlines(self):
        text = "AAPL beats earnings estimates and raises guidance"
        res = self.analyzer.analyze(text)
        self.assertGreater(res.polarity, 0.3)
        self.assertGreater(res.confidence, 0.5)
        self.assertIn("earnings_beat", res.events)
        self.assertIn("guidance_raise", res.events)

    def test_negative_headlines(self):
        text = "TSLA shares plunge as it files for bankruptcy"
        res = self.analyzer.analyze(text)
        self.assertLess(res.polarity, -0.5)
        self.assertIn("bankruptcy", res.events)

    def test_negation_handling(self):
        text1 = "Company reports profit"
        text2 = "Company reports no profit"
        
        res1 = self.analyzer.analyze(text1)
        res2 = self.analyzer.analyze(text2)
        
        self.assertGreater(res1.polarity, 0.0)
        self.assertLess(res2.polarity, 0.0)


class TestSentimentDecisionEngine(unittest.TestCase):
    def setUp(self):
        self.engine = SentimentDecisionEngine(market="us")
        # Override weights and thresholds for deterministic testing
        self.engine.enabled = True
        self.engine.min_confidence = 0.5
        self.engine.strong_neg_threshold = -0.5
        self.engine.w_signal = 0.4
        self.engine.w_news = 0.4
        self.engine.w_market = 0.1
        self.engine.w_sector = 0.1

    @patch("sentiment.engine.NewsCollector")
    def test_evaluate_confirm_and_reject(self, mock_collector_cls):
        # Setup mock articles
        mock_collector = MagicMock()
        mock_collector.stats = {"sources_used": ["Yahoo"], "duplicates_removed": 0}
        
        now = datetime.now(timezone.utc)
        articles = [
            # Positive news for AAPL
            Article(headline="AAPL beats earnings and upgrades guidance", published=now - timedelta(hours=1), ticker="AAPL", url="http://aapl1"),
            # Strong negative news for TSLA
            Article(headline="TSLA files for bankruptcy", published=now - timedelta(hours=1), ticker="TSLA", url="http://tsla1"),
            # Market news
            Article(headline="Stocks rally as Fed interest rate cut looks likely", published=now - timedelta(hours=2), ticker="", url="http://mkt1")
        ]
        mock_collector.fetch.return_value = articles
        self.engine.collector = mock_collector
        
        picks = [
            {"ticker": "AAPL", "score": 1.5},  # Z-score of 1.5 -> prob ~ 0.93
            {"ticker": "TSLA", "score": 2.0}   # Z-score of 2.0 -> prob ~ 0.97 but will get rejected due to strong negative sentiment
        ]
        
        confirmed, report = self.engine.evaluate(picks, datetime.now())
        
        # Verify results
        self.assertEqual(len(confirmed), 1)
        self.assertEqual(confirmed[0]["ticker"], "AAPL")
        
        # Verify stats in report
        stats = report["stats"]
        self.assertEqual(stats["confirmed"], 1)
        self.assertEqual(stats["rejected"], 1)
        self.assertEqual(stats["held"], 0)


if __name__ == "__main__":
    unittest.main()
