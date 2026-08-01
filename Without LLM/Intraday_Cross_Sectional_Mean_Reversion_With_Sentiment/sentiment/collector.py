"""
sentiment/collector.py
======================
Multi-source news collector with parallel fetching, deduplication, and caching.

Sources:
    - Yahoo Finance RSS (free, no auth)
    - Google News RSS (free, no auth)
    - Alpaca News API (optional, requires subscription)

Architecture:
    - All sources implement the NewsSource interface
    - NewsCollector orchestrates parallel fetching via ThreadPoolExecutor
    - Articles are deduplicated by URL hash and cached with configurable TTL
    - Graceful fallback: if any source fails, others continue

Usage:
    collector = NewsCollector(sources=["yahoo_rss", "google_rss"])
    articles = collector.fetch(tickers=["AAPL", "TSLA"], market_news=True)
"""

from __future__ import annotations

import hashlib
import logging
import time
import re
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError
from xml.etree import ElementTree as ET
from html import unescape

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data Model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Article:
    """Represents a single news article."""
    headline: str
    summary: str = ""
    published: Optional[datetime] = None
    source: str = ""
    ticker: str = ""           # Primary ticker mentioned (empty = market-wide)
    url: str = ""
    category: str = ""         # e.g. "earnings", "macro", "analyst", "general"
    
    @property
    def url_hash(self) -> str:
        """Unique hash for deduplication."""
        key = self.url if self.url else f"{self.headline}:{self.source}"
        return hashlib.md5(key.encode()).hexdigest()
    
    @property
    def age_hours(self) -> float:
        """Age in hours from now. Returns 999 if no publish time."""
        if not self.published:
            return 999.0
        now = datetime.now(timezone.utc)
        pub = self.published if self.published.tzinfo else self.published.replace(tzinfo=timezone.utc)
        delta = now - pub
        return max(0.0, delta.total_seconds() / 3600.0)


def _clean_html(raw: str) -> str:
    """Strip HTML tags and unescape entities."""
    text = re.sub(r"<[^>]+>", " ", raw)
    text = unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def _safe_fetch_url(url: str, timeout: int = 10) -> Optional[str]:
    """Fetch URL content with timeout and error handling."""
    try:
        req = Request(url, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })
        with urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except (URLError, HTTPError, OSError, Exception) as e:
        log.debug("Failed to fetch %s: %s", url, str(e)[:100])
        return None


def _parse_rfc822_date(date_str: str) -> Optional[datetime]:
    """Parse RFC 822 date format commonly used in RSS feeds."""
    formats = [
        "%a, %d %b %Y %H:%M:%S %z",
        "%a, %d %b %Y %H:%M:%S %Z",
        "%a, %d %b %Y %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%d %H:%M:%S",
    ]
    date_str = date_str.strip()
    # Remove named timezone like "GMT", "EST" if no offset
    date_cleaned = re.sub(r"\s+(GMT|UTC|EST|EDT|CST|CDT|PST|PDT)$", "", date_str)
    
    for fmt in formats:
        try:
            dt = datetime.strptime(date_cleaned, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            continue
    return None


# ─────────────────────────────────────────────────────────────────────────────
# News Source Interface
# ─────────────────────────────────────────────────────────────────────────────

class NewsSource(ABC):
    """Abstract base for all news sources."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable source name."""
        ...
    
    @abstractmethod
    def fetch_ticker(self, ticker: str, max_age_hours: float = 24) -> list[Article]:
        """Fetch articles for a specific ticker."""
        ...
    
    @abstractmethod
    def fetch_market(self, market: str = "us", max_age_hours: float = 24) -> list[Article]:
        """Fetch general market/economy news."""
        ...


# ─────────────────────────────────────────────────────────────────────────────
# Yahoo Finance RSS
# ─────────────────────────────────────────────────────────────────────────────

class YahooRSSSource(NewsSource):
    """
    Fetches news from Yahoo Finance RSS feed.
    Free, no authentication required.
    """
    
    BASE_URL = "https://feeds.finance.yahoo.com/rss/2.0/headline"
    
    @property
    def name(self) -> str:
        return "Yahoo Finance"
    
    def fetch_ticker(self, ticker: str, max_age_hours: float = 24) -> list[Article]:
        url = f"{self.BASE_URL}?s={ticker}&region=US&lang=en-US"
        return self._parse_feed(url, ticker=ticker, max_age_hours=max_age_hours)
    
    def fetch_market(self, market: str = "us", max_age_hours: float = 24) -> list[Article]:
        # Yahoo Finance market news via ^GSPC (S&P 500) and ^IXIC (NASDAQ)
        articles = []
        for symbol in ["^GSPC", "^IXIC", "^DJI"]:
            url = f"{self.BASE_URL}?s={symbol}&region=US&lang=en-US"
            articles.extend(self._parse_feed(url, ticker="", max_age_hours=max_age_hours))
        return articles
    
    def _parse_feed(self, url: str, ticker: str = "", max_age_hours: float = 24) -> list[Article]:
        xml_text = _safe_fetch_url(url, timeout=8)
        if not xml_text:
            return []
        
        articles = []
        try:
            root = ET.fromstring(xml_text)
            for item in root.findall(".//item"):
                headline = item.findtext("title", "").strip()
                if not headline:
                    continue
                
                summary = _clean_html(item.findtext("description", ""))
                pub_date = _parse_rfc822_date(item.findtext("pubDate", ""))
                link = item.findtext("link", "")
                
                art = Article(
                    headline=headline,
                    summary=summary,
                    published=pub_date,
                    source=self.name,
                    ticker=ticker,
                    url=link,
                    category="general",
                )
                
                if art.age_hours <= max_age_hours:
                    articles.append(art)
                    
        except ET.ParseError as e:
            log.debug("Yahoo RSS parse error for %s: %s", ticker, e)
        
        return articles


# ─────────────────────────────────────────────────────────────────────────────
# Google News RSS
# ─────────────────────────────────────────────────────────────────────────────

class GoogleNewsRSSSource(NewsSource):
    """
    Fetches news from Google News RSS.
    Free, no authentication required.
    """
    
    BASE_URL = "https://news.google.com/rss/search"
    
    @property
    def name(self) -> str:
        return "Google News"
    
    def fetch_ticker(self, ticker: str, max_age_hours: float = 24) -> list[Article]:
        query = f"{ticker}+stock+news"
        url = f"{self.BASE_URL}?q={query}&hl=en-US&gl=US&ceid=US:en"
        return self._parse_feed(url, ticker=ticker, max_age_hours=max_age_hours)
    
    def fetch_market(self, market: str = "us", max_age_hours: float = 24) -> list[Article]:
        queries = [
            "stock+market+today",
            "federal+reserve+economy",
            "nasdaq+market+news",
        ]
        articles = []
        for q in queries:
            url = f"{self.BASE_URL}?q={q}&hl=en-US&gl=US&ceid=US:en"
            articles.extend(self._parse_feed(url, ticker="", max_age_hours=max_age_hours))
        return articles
    
    def _parse_feed(self, url: str, ticker: str = "", max_age_hours: float = 24) -> list[Article]:
        xml_text = _safe_fetch_url(url, timeout=8)
        if not xml_text:
            return []
        
        articles = []
        try:
            root = ET.fromstring(xml_text)
            for item in root.findall(".//item"):
                headline = _clean_html(item.findtext("title", "")).strip()
                if not headline:
                    continue
                
                summary = _clean_html(item.findtext("description", ""))
                pub_date = _parse_rfc822_date(item.findtext("pubDate", ""))
                link = item.findtext("link", "")
                source_name = item.findtext("source", self.name)
                
                art = Article(
                    headline=headline,
                    summary=summary,
                    published=pub_date,
                    source=f"Google/{source_name}",
                    ticker=ticker,
                    url=link,
                    category="general",
                )
                
                if art.age_hours <= max_age_hours:
                    articles.append(art)
                    
        except ET.ParseError as e:
            log.debug("Google News RSS parse error: %s", e)
        
        return articles


# ─────────────────────────────────────────────────────────────────────────────
# Alpaca News API (Optional)
# ─────────────────────────────────────────────────────────────────────────────

class AlpacaNewsSource(NewsSource):
    """
    Fetches news from Alpaca News API.
    Requires Alpaca API credentials and a news subscription.
    """
    
    def __init__(self, api_key: str = "", secret_key: str = ""):
        self._client = None
        if api_key and secret_key:
            try:
                from alpaca.data.news import NewsClient
                from alpaca.data.requests import NewsRequest
                self._client = NewsClient(api_key, secret_key)
                self._NewsRequest = NewsRequest
            except (ImportError, Exception) as e:
                log.debug("Alpaca News API not available: %s", e)
    
    @property
    def name(self) -> str:
        return "Alpaca News"
    
    @property
    def available(self) -> bool:
        return self._client is not None
    
    def fetch_ticker(self, ticker: str, max_age_hours: float = 24) -> list[Article]:
        if not self.available:
            return []
        
        try:
            start = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
            request = self._NewsRequest(
                symbols=[ticker],
                start=start,
                limit=20,
                sort="desc",
            )
            news = self._client.get_news(request)
            return [self._convert(n, ticker) for n in news.news if n.headline]
        except Exception as e:
            log.debug("Alpaca News fetch failed for %s: %s", ticker, str(e)[:100])
            return []
    
    def fetch_market(self, market: str = "us", max_age_hours: float = 24) -> list[Article]:
        if not self.available:
            return []
        
        try:
            start = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
            request = self._NewsRequest(
                start=start,
                limit=30,
                sort="desc",
            )
            news = self._client.get_news(request)
            return [self._convert(n, "") for n in news.news if n.headline]
        except Exception as e:
            log.debug("Alpaca market news fetch failed: %s", str(e)[:100])
            return []
    
    def _convert(self, news_item, ticker: str) -> Article:
        return Article(
            headline=news_item.headline,
            summary=getattr(news_item, "summary", ""),
            published=getattr(news_item, "created_at", None),
            source=self.name,
            ticker=ticker,
            url=getattr(news_item, "url", ""),
            category="general",
        )


# ─────────────────────────────────────────────────────────────────────────────
# News Collector (Orchestrator)
# ─────────────────────────────────────────────────────────────────────────────

class NewsCollector:
    """
    Orchestrates parallel news collection from multiple sources.
    
    Features:
        - Parallel fetching via ThreadPoolExecutor
        - Deduplication by URL hash
        - Caching with configurable TTL
        - Graceful error handling per-source
    
    Parameters
    ----------
    sources : list[str]
        Source names to enable: "yahoo_rss", "google_rss", "alpaca"
    max_age_hours : float
        Maximum article age in hours (default: 24)
    max_workers : int
        Thread pool size for parallel fetching (default: 8)
    cache_ttl_seconds : int
        Cache time-to-live in seconds (default: 300 = 5 minutes)
    alpaca_api_key : str
        Alpaca API key (optional, for Alpaca News source)
    alpaca_secret_key : str
        Alpaca secret key (optional, for Alpaca News source)
    """
    
    def __init__(
        self,
        sources: list[str] = None,
        max_age_hours: float = 24,
        max_workers: int = 8,
        cache_ttl_seconds: int = 300,
        alpaca_api_key: str = "",
        alpaca_secret_key: str = "",
    ):
        self.max_age_hours = max_age_hours
        self.max_workers = max_workers
        self.cache_ttl = cache_ttl_seconds
        
        # Build active sources
        source_names = sources or ["yahoo_rss", "google_rss"]
        self._sources: list[NewsSource] = []
        
        for name in source_names:
            if name == "yahoo_rss":
                self._sources.append(YahooRSSSource())
            elif name == "google_rss":
                self._sources.append(GoogleNewsRSSSource())
            elif name == "alpaca":
                src = AlpacaNewsSource(alpaca_api_key, alpaca_secret_key)
                if src.available:
                    self._sources.append(src)
                else:
                    log.info("[SENTIMENT] Alpaca News API not available, skipping")
        
        # Cache: {cache_key: (timestamp, articles)}
        self._cache: dict[str, tuple[float, list[Article]]] = {}
        
        # Stats
        self.stats = {
            "total_fetched": 0,
            "duplicates_removed": 0,
            "sources_failed": 0,
            "sources_used": [s.name for s in self._sources],
        }
    
    def fetch(
        self,
        tickers: list[str],
        market_news: bool = True,
        market: str = "us",
    ) -> list[Article]:
        """
        Fetch news for all tickers and optionally market-wide news.
        
        Returns deduplicated, age-filtered list of Articles.
        """
        t0 = time.perf_counter()
        all_articles: list[Article] = []
        seen_hashes: set[str] = set()
        duplicates = 0
        
        # Check cache
        cache_key = f"{','.join(sorted(tickers))}:{market}:{market_news}"
        if cache_key in self._cache:
            ts, cached = self._cache[cache_key]
            if time.time() - ts < self.cache_ttl:
                log.debug("[SENTIMENT] Using cached articles (%d)", len(cached))
                return cached
        
        # Build fetch tasks
        tasks = []
        for source in self._sources:
            if market_news:
                tasks.append(("market", source, market, ""))
            for ticker in tickers:
                tasks.append(("ticker", source, "", ticker))
        
        # Execute in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {}
            for task_type, source, mkt, ticker in tasks:
                if task_type == "market":
                    fut = pool.submit(source.fetch_market, mkt, self.max_age_hours)
                else:
                    fut = pool.submit(source.fetch_ticker, ticker, self.max_age_hours)
                futures[fut] = (source.name, task_type, ticker or mkt)
            
            for fut in as_completed(futures, timeout=30):
                src_name, task_type, label = futures[fut]
                try:
                    articles = fut.result(timeout=10)
                    for art in articles:
                        h = art.url_hash
                        if h not in seen_hashes:
                            seen_hashes.add(h)
                            all_articles.append(art)
                        else:
                            duplicates += 1
                except Exception as e:
                    self.stats["sources_failed"] += 1
                    log.debug("[SENTIMENT] %s failed for %s: %s", src_name, label, str(e)[:100])
        
        # Update stats
        self.stats["total_fetched"] = len(all_articles)
        self.stats["duplicates_removed"] = duplicates
        
        # Cache result
        self._cache[cache_key] = (time.time(), all_articles)
        
        elapsed = time.perf_counter() - t0
        log.info("[SENTIMENT] Collected %d articles (%d duplicates removed) in %.1fs from %s",
                 len(all_articles), duplicates, elapsed,
                 ", ".join(s.name for s in self._sources))
        
        return all_articles
