# 📈 Intraday Cross-Sectional Mean Reversion (NASDAQ / US Market)

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A professional-grade, quantitative **intraday mean reversion trading strategy** built for the US Equity markets (NASDAQ/NYSE). 

This repository provides a complete, autonomous pipeline that dynamically screens the US market universe, calculates alpha features from 1-minute to 15-minute OHLCV data, models signals using Information Coefficient (IC) weighting, and constructs a dollar-neutral, risk-managed portfolio.

---

## 📋 Project Documentation

Detailed tracking of every architectural decision and version update is maintained in our Decision Records:
👉 **[View CHANGELOG_ADR.md](./CHANGELOG_ADR.md)**

---

## ✨ Features

- **Dynamic US Universe Selection:** Auto-screens volatile tech and growth stocks from a master pool using rolling ATR% and liquidity rankings.
- **Robust Data Pipeline:** Built-in `yfinance` integration for US tickers. Automatically aligns strictly to US market hours (9:30 AM - 4:00 PM EST).
- **7-Factor Alpha Engine:** Computes sophisticated cross-sectional features including Bar Reversals, VWAP Deviations, Volume Shocks, and Mean Reversion z-scores.
- **Smart Portfolio Construction:** Generates beta-neutral, volatility-scaled position weights while factoring in transaction costs and target leverage.
- **Strict Risk Management:** Market-wide stress detection (leveraging QQQ/SPY correlations) to dynamically scale down exposure during extreme volatility.
- **Pure Intraday Execution:** Constraints ensured for single-day trades only, eliminating overnight gap risk (v0.2.0 optimization).

---

## 🔬 Detailed Strategy Breakdown

This system implements a strictly systematic, cross-sectional mean-reversion strategy. It operates on the premise that large intraday idiosyncratic price moves are often overextended and likely to revert toward their local average.

### 1. The 7-Factor Alpha Engine
The strategy computes 7 distinct alpha features for every stock in the universe at every bar:

| Feature | Logic | Market Hypothesis |
| :--- | :--- | :--- |
| **Bar Reversal** | Single-bar price change vs ATR | High-velocity moves often represent liquidity gaps that fill. |
| **Short Rev** | 15-min momentum fade | Short-term trend exhaustion in high-frequency data. |
| **VWAP Dev** | Deviation from session VWAP | Institutional "anchoring" effect; price reverts to the volume-weighted mean. |
| **Vol Shock** | Relative volume spike vs TOD | Volume spikes often signal the end of a move rather than a continuation. |
| **Vol Burst** | High range opposite to bar close | Buyer/Seller exhaustion; the final "push" before a reversal. |
| **Residual Return**| Market-beta adjusted returns | Isolates stock-specific noise from broad market moves (SPY). |
| **Mkt Exhaustion** | Move vs Avg Intraday Range | Stocks exceeding their typical daily range lose momentum. |

### 2. Adaptive Signal Combination (Spearman IC)
Instead of static weights, the model uses a **Dynamic IC-Weighting** scheme:
- **Rolling IC Ranking**: The system calculates the Spearman Rank Correlation (Information Coefficient) between each feature and the subsequent price move over a rolling window.
- **Self-Correcting Weights**: Features that are currently "predictive" get higher weights. Features that have lost their edge (low t-stat) are automatically zeroed out, allowing the strategy to adapt to different market regimes.

### 3. Portfolio Construction & Neutralization
- **Volatility Equalization**: Positions are **Risk-Weighted**. Each stock's weight is scaled by its inverse realized volatility (EWM), ensuring that "calm" stocks and "wild" stocks contribute equal risk.
- **Beta/Dollar Neutrality**: The portfolio is rebalanced to maintain a zero-net exposure. For every dollar long, there is a dollar short. This minimizes exposure to broad market moves.
- **Weight Smoothing**: To prevent excessive trading costs, weights are smoothed using an EWM decay, ensuring the strategy only trades on persistent signals.

### 4. Professional-Grade Risk Guardrails
- **Extreme Move Scaling**: If a stock moves > 8% intraday, the position is halved immediately to mitigate "black swan" risk.
- **Market Stress Kill-Switch**: If the underlying index (QQQ) moves > 3% in a short window, the entire portfolio leverage is reduced to 50%.
- **Drawdown-Adaptive Sizing**: The system monitors its own equity curve; as drawdowns increase, it linearly scales down total exposure to protect capital.

---

## 🚀 Quick Start (UX Friendly Guide)

### 1. Prerequisites
You only need Python 3.9+ and pip installed. We recommend using a virtual environment (`.venv`).

### 2. Installation
Clone the repo and install the required dependencies:
```bash
git clone https://github.com/yourusername/us-mean-reversion.git
cd us-mean-reversion
pip install -r requirements.txt
```

### 3. Run the Strategy

**A. The "Test Drive" (Fastest)**
Want to see how it works instantly? Run it on just 3 high-volume tech stocks for the last 30 days:
```bash
python run_pipeline.py --tickers NVDA TSLA AAPL --days 30
```

**B. The "Full Production Run"**
Execute the entire pipeline: dynamically screen the universe, fetch data, compute alpha, and simulate portfolio returns:
```bash
python run_pipeline.py
```

---

## 🧠 Architecture Overview

The codebase is modular and designed for easy extension by quants and developers:

- **`config/config.yaml`**: The brain of the operation. Change leverage, capital (default $100,000), transaction costs, and signal windows here.
- **`features/`**: The math layer. Transforms raw OHLCV into cross-sectional Z-scores. 
- **`alpha/`**: The portfolio layer. Weights the 7 features based on their rolling predictive power (IC), and outputs target trade weights.
- **`reports/`**: Detailed backtesting logs and performance metrics.

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

> **Disclaimer:** This project is for educational and research purposes only. It is not financial advice. Trading equities on margin involves significant risk. MS-Market hours and volatility behavior differ significantly from other exchanges.