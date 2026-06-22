"""
Unit tests for the Claude-free enhancements:

  * new strategies (breakout, VWAP reversion)
  * news feed (graceful degradation + headline filtering)
  * sentiment risk gate (veto / observe / TTL cache)

All tests run offline — any network access is mocked — so they are safe and
fast. Run with::

    venv/Scripts/python.exe -m pytest test_enhancements.py -q
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import numpy as np
import pandas as pd

from src.data.news_feed import NewsFeed
from src.risk.sentiment_gate import SentimentGate
from src.strategy.base import Signal, TradeAction
from src.strategy.breakout import BreakoutStrategy
from src.strategy.vwap_reversion import VWAPReversionStrategy


def _run(coro):
    return asyncio.run(coro)


def _base_frame(seed: int = 1, n: int = 200):
    rng = np.random.default_rng(seed)
    price = 100 + np.cumsum(rng.standard_normal(n)) * 0.5
    vol = rng.integers(800, 1200, n).astype(float)
    return price, vol


# ----------------------------------------------------------------- strategies

def test_breakout_emits_buy_on_range_break():
    price, vol = _base_frame()
    price[-1] = price[:-1].max() + 5
    vol[-1] = 6000.0
    df = pd.DataFrame(
        {"open": price, "high": price + 1, "low": price - 1, "close": price, "volume": vol}
    )
    df.attrs["symbol"] = "TEST/USDT"

    strat = BreakoutStrategy()
    sig = strat.analyze(df)
    assert sig.action == TradeAction.BUY
    assert strat.should_enter(df, sig) is True
    assert sig.stop_loss_price < price[-1] < sig.take_profit_price


def test_vwap_reversion_emits_buy_on_dip():
    price, vol = _base_frame(seed=2)
    price[-1] = price[-2] * 0.95  # stretched well below recent VWAP
    vol[-1] = 3000.0
    df = pd.DataFrame(
        {"open": price, "high": price + 1, "low": price - 1, "close": price, "volume": vol}
    )
    df.attrs["symbol"] = "TEST/USDT"

    strat = VWAPReversionStrategy()
    sig = strat.analyze(df)
    assert sig.action == TradeAction.BUY
    assert strat.should_enter(df, sig) is True


def test_strategies_hold_on_short_frame():
    df = pd.DataFrame(
        {"open": [1.0] * 10, "high": [1.0] * 10, "low": [1.0] * 10,
         "close": [1.0] * 10, "volume": [1.0] * 10}
    )
    df.attrs["symbol"] = "X/USDT"
    assert BreakoutStrategy().analyze(df).action == TradeAction.HOLD
    assert VWAPReversionStrategy().analyze(df).action == TradeAction.HOLD


# ----------------------------------------------------------------- news feed

def test_stock_news_empty_without_keys():
    feed = NewsFeed()  # no alpaca creds
    assert _run(feed.fetch_stock_news("AAPL")) == []


def test_rss_filters_by_coin_alias(monkeypatch):
    feed = NewsFeed()
    titles = [
        "Bitcoin hits new high",
        "Dogecoin rallies on hype",
        "Some unrelated market note",
        "DOGE whales accumulate",
    ]

    async def fake_fetch_one(_session, _url):
        return titles

    monkeypatch.setattr(NewsFeed, "_fetch_one_rss", staticmethod(fake_fetch_one))
    out = _run(feed._fetch_rss("DOGE", limit=10))
    # Only dogecoin/doge headlines should match the alias filter.
    assert out
    assert all("doge" in t.lower() for t in out)


def test_alpaca_headline_extraction():
    class _Art:
        def __init__(self, h):
            self.headline = h

    class _NewsSet:
        data = {"AAPL": [_Art("Apple beats earnings"), _Art("New iPhone")]}

    out = NewsFeed._extract_alpaca_headlines(_NewsSet(), limit=10)
    assert out == ["Apple beats earnings", "New iPhone"]


# ------------------------------------------------------------- sentiment gate

def _signal(action=TradeAction.BUY):
    return Signal(
        symbol="DOGE/USDT", action=action, confidence=0.6,
        stop_loss_price=1.0, take_profit_price=2.0, strategy_name="mean_reversion",
    )


def test_gate_vetoes_buy_into_bearish_when_enforcing():
    gate = SentimentGate(threshold=0.5, enforce=True)
    gate._news.fetch_news = AsyncMock(return_value=["panic dump crash"])
    gate._analyzer.get_market_sentiment = AsyncMock(return_value=-0.8)

    sig = _signal(TradeAction.BUY)
    allow, _reason, score = _run(gate.check(sig))
    assert allow is False
    assert score == -0.8
    assert sig.metadata.get("sentiment") == -0.8


def test_gate_observe_mode_allows_but_annotates():
    gate = SentimentGate(threshold=0.5, enforce=False)
    gate._news.fetch_news = AsyncMock(return_value=["panic dump crash"])
    gate._analyzer.get_market_sentiment = AsyncMock(return_value=-0.9)

    sig = _signal(TradeAction.BUY)
    allow, reason, _score = _run(gate.check(sig))
    assert allow is True                       # observe never blocks
    assert "observe" in reason
    assert sig.metadata.get("sentiment") == -0.9


def test_gate_allows_aligned_sentiment():
    gate = SentimentGate(threshold=0.5, enforce=True)
    gate._news.fetch_news = AsyncMock(return_value=["bullish rally moon"])
    gate._analyzer.get_market_sentiment = AsyncMock(return_value=0.7)

    allow, _reason, _score = _run(gate.check(_signal(TradeAction.BUY)))
    assert allow is True


def test_gate_caches_score_within_ttl():
    gate = SentimentGate(threshold=0.5, enforce=True, cache_ttl=1800)
    gate._news.fetch_news = AsyncMock(return_value=["x"])
    gate._analyzer.get_market_sentiment = AsyncMock(return_value=-0.8)

    first = _run(gate.score("DOGE/USDT"))
    # Change the underlying mock; cached value must still be returned.
    gate._analyzer.get_market_sentiment = AsyncMock(return_value=0.9)
    second = _run(gate.score("DOGE/USDT"))
    assert first == second == -0.8
    gate._analyzer.get_market_sentiment.assert_not_called()
