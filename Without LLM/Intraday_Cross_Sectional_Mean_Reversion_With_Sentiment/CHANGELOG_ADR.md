# Architecture & Version Decision Record

This document maintains a chronological formal record of significant changes, architectural decisions (ADRs), and version updates. This structure ensures that we capture the "why" (context) and the "when" behind our design choices, not just the "what."

## 📋 Version Index

| Version | Date | Time | Change Summary | Status | Details |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **v1.2.2** | 2026-06-24 | 00:49:00 | Fix short position flattening logic in live pipeline | `Active` | [View Details](#v122---fix-short-position-flattening-logic-in-live-pipeline) |
| **v1.2.1** | 2026-06-23 | 21:21:00 | Fix yfinance lookback data limit for live trading | `Active` | [View Details](#v121---fix-yfinance-lookback-data-limit-for-live-trading) |
| **v1.2.0** | 2026-04-18 | 23:54:00 | Documentation Sync & ADR Formalization | `Active` | [View Details](#v120---documentation-sync--adr-formalization) |
| **v1.1.0** | 2026-04-16 | 19:54:29 | Optimized NASDAQ pipeline execution parameters | `Active` | [View Details](#v110---optimized-nasdaq-pipeline-execution-parameters) |
| **v1.0.1** | 2026-04-15 | 15:35:17 | Established version tracking and ADR template | `Active` | [View Details](#v101---established-version-tracking-and-adr-template) |
| **v1.0.0** | 2026-04-15 | 10:00:00 | Migrated trading pipeline from NSE to US Market | `Active` | [View Details](#v100---migrated-trading-pipeline-from-nse-to-us-market) |
| **v0.2.0** | 2026-04-14 | 20:10:55 | Enforced single day trades constraint | `Deprecated` | [View Details](#v020---enforced-single-day-trades-constraint) |
| **v0.1.0** | 2026-03-26 | 19:52:38 | Initial implementation & 15-minute strategy baseline | `Deprecated` | [View Details](#v010---initial-implementation--15-minute-strategy-baseline) |

---

## 📝 Detailed Version Logs

### v1.2.2 - Fix short position flattening logic in live pipeline

* **Date:** 2026-06-24
* **Time:** 00:49:00 (+05:30)

#### 1. Context & Problem Statement
During a previous session, a double shutdown/restart fallback caused the execution engine to double-sell several long positions (specifically `LAMR`, `LITE`, and `VIAV`), which created short positions on the Alpaca broker account. The subsequent flattening logic in `run_live.py` (both in `graceful_shutdown` and the main monitoring cycle) was hardcoded to execute `OrderSide.SELL` for any position returned by the broker. Since `execute_order` rejects negative quantities, the short positions were never closed and remained active in the account across sessions.

#### 2. Decision / Changes Implemented
* Modified `graceful_shutdown` and the `FLATTEN PHASE` in `run_universe_session` in `run_live.py` to inspect the broker position's quantity (`pos.qty`):
  - If `qty > 0` (long position), submit a `SELL` order for `qty`.
  - If `qty < 0` (short position), submit a `BUY` order for `abs(qty)`.
* Executed manual buy cover orders to flatten the existing short positions (`LAMR` 8, `LITE` 5, `VIAV` 105) in the broker.

#### 3. Consequences
* **Positive:** Both long and short positions will now be correctly and cleanly closed at the end of each session or during manual aborts, preventing stale positions from carrying over.
* **Risks:** None identified.

<br/><hr/><br/>

### v1.2.1 - Fix yfinance lookback data limit for live trading

* **Date:** 2026-06-23
* **Time:** 21:21:00 (+05:30)

#### 1. Context & Problem Statement
The morning entry phase in the live trading pipeline fetches 35 days of history. However, `fetch_yfinance_panels` was mapping any lookback > 5 to the `"1mo"` period. Because of weekends and holidays, 1 month of 5-minute bars often only provides 19 trading days of data. This caused a silent execution suspension when `DailySignalEngine` validated history against its 20-day lookback requirement, rejecting candidate lists and stalling trades.

#### 2. Decision / Changes Implemented
* Modified `fetch_yfinance_panels` inside `run_live.py` to check `lookback_days` dynamically:
  - If `lookback_days <= 5`, use `"5d"`.
  - If `5 < lookback_days <= 20`, use `"1mo"`.
  - If `lookback_days > 20`, use `"60d"` (within the 60-day limit for 5m intraday bars).
* Verified that a dry run successfully retrieves history, computes composite z-scores, and generates stock pick targets.

#### 3. Consequences
* **Positive:** Live execution will robustly acquire 60 days of 5-minute bar history, preventing insufficient history exceptions and ensuring daily trade selections occur reliably.
* **Risks:** The 60-day limit is a hard cap set by Yahoo Finance for 5m bars. If a lookback of >60 days is requested, the API call will fail. Historical backtests longer than 60 days must rely on local Parquet databases via the Polygon fetch script.

<br/><hr/><br/>

### v1.2.0 - Documentation Sync & ADR Formalization

* **Date:** 2026-04-18
* **Time:** 23:54:00 (+05:30)

#### 1. Context & Problem Statement
Synchronized all quant models and architectural designs with official developer onboarding walkthroughs to ensure clarity for new team members.

#### 2. Decision / Changes Implemented
* Created the comprehensive Developer & Quant Onboarding Guide (`DEVELOPER_ONBOARDING.md`).
* Formally documented version logs up to `v1.2.0`.

#### 3. Consequences
* **Positive:** Developers onboarding have a complete reference point for the US Market migration, alpha models, and playbook simulations.

<br/><hr/><br/>

### v1.1.0 - Optimized NASDAQ pipeline execution parameters

* **Date:** 2026-04-16
* **Time:** 19:54:29 (+05:30)

#### 1. Context & Problem Statement
Following the migration to the US market, the strategy was experiencing frequent "Time Exit" trades and suboptimal profit capture. Execution parameters needed calibration to maximize daily profitability and improve the balance between risk management and profit capture.

#### 2. Decision / Changes Implemented
* Calibrated execution parameters including stop-loss, profit-taking, and trailing stop triggers for the NASDAQ market.
* Refined intraday mean-reversion logic to better capture market anomalies.
* Maintained a robust, automated logging and backtesting workflow.

#### 3. Consequences
* **Positive:** Intraday trades now have an optimized exit strategy, reducing "Time Exit" frequency. Increased daily profitability margins through improved handling of risk parameters.
* **Risks:** The calibrated parameters are specifically tuned for the current market regime and might need periodic adjustments depending on volatility.

<br/><hr/><br/>

### v1.0.1 - Established version tracking and ADR template

* **Date:** 2026-04-15
* **Time:** 15:35:17 (+05:30)

#### 1. Context & Problem Statement
As the project scales and multiple trading strategies are iteratively improved, a professional, centralized document is required to maintain the history of all changes. It is crucial to have a source of truth—inspired by Arc42 and standard ADRs—that efficiently tracks version names, exact timestamps, and detailed rationales.

#### 2. Decision / Changes Implemented
* Created the `CHANGELOG_ADR.md` document at the project root.
* Implemented a tabular index at the top, equipped with local markdown references linking directly to the comprehensive version elaborations.
* Mandated a structured layout containing **Context**, **Decision**, and **Consequences** for future architectural updates.

#### 3. Consequences
* **Positive:** Greatly improves maintainability, simplifies onboarding, and clearly documents the evolution of the quantitative strategies.
* **Risks:** The log requires manual updating, so it must be added to the standard development lifecycle following any major milestone.

<br/><hr/><br/>

### v1.0.0 - Migrated trading pipeline from NSE to US Market

* **Date:** 2026-04-15
* **Time:** 10:00:00 (+05:30)

#### 1. Context & Problem Statement
The original intraday mean-reversion trading pipeline was tailored exclusively for the Indian Stock Exchange (NSE). In order to capture different market anomalies, the decision was made to reconfigure the system entirely for the US market.

#### 2. Decision / Changes Implemented
* Systematically removed all NSE-specific logic including session times, local circuit breaker rules, and currency symbols.
* Reconfigured the target portfolio for a 80-stock US tech/growth strategy, starting with $100,000 capital.
* Switched the primary hedging component to QQQ.

#### 3. Consequences
* The pipeline is fully equipped to backtest against US market hours correctly.
* Any lingering dependencies on the previous feature engine meant strictly for NSE have been deprecated.

<br/><hr/><br/>

### v0.2.0 - Enforced single day trades constraint

* **Date:** 2026-04-14
* **Time:** 20:10:55 (+05:30)

#### 1. Context & Problem Statement
The strategy previously held positions across multiple sessions which introduced overnight gap risks. It was necessary to strictly restrict the pipeline to execute and close all trades within a single trading day to validate specific mean reversion hypothesis.

#### 2. Decision / Changes Implemented
* Implemented strict intraday constraints ensuring all positions are squared off by the end of the trading session.
* Refactored trading logic to prioritize single day trades.

#### 3. Consequences
* **Positive:** Reduced exposure to overnight market risk. Transformed the algorithm into a pure intraday trading system.
* **Risks:** Missing out on potential multi-day continued momentum.

<br/><hr/><br/>

### v0.1.0 - Initial implementation & 15-minute strategy baseline

* **Date:** 2026-03-26
* **Time:** 19:52:38 (+05:30)

#### 1. Context & Problem Statement
Initial conceptualization and implementation of the Intraday Cross-Sectional Mean Reversion strategy.

#### 2. Decision / Changes Implemented
* Established the initial quantitative strategy operating on 15-minute interval data.
* Set up the fundamental backtesting pipeline for evaluating strategy performance.
* Implemented baseline logic for signal generation and position sizing.

#### 3. Consequences
* **Positive:** Provided a functional baseline for subsequent metric tracking and logical improvements. Required infrastructure for execution created.
* **Risks:** Baseline strategy needed significant tuning since early assumptions heavily affect future behavior.
