# NEXUS Core: Advanced Algorithmic Trading Engine
## Comprehensive Technical Documentation & Architecture Reference
**Version:** 2.1.0 ("Full Might" Evolution) | **Target Asset:** XAUUSDm (Gold)

---

## Executive Summary
NEXUS Core is an advanced, multi-threaded algorithmic trading engine designed specifically for Gold CFD (XAUUSDm) markets. It operates as a hybrid quantitative system, combining traditional Smart Money Concepts (SMC) structure scanning with a Machine Learning (RandomForest) "Two-Key" momentum filter. This document outlines the deep technical architecture, mathematical models, risk management engines, and the Active Learning (Evolution) systems that govern the bot's autonomy.

---

## 1. Architectural Overview

### 1.1 System Topography
The system is cleanly decoupled into three primary nodes:
1. **The Orchestrator (`goldvx.py`)**: The central nervous system. It handles MT5 connection persistence, global state management, continuous data polling, and strategy multiplexing.
2. **The Logic Hub (`strategies/`)**: Contains pure mathematical and structural logic (SMC, VWAP, BB Breakouts). It analyzes Pandas DataFrames and returns raw payloads without any direct connection to the broker.
3. **The Execution Engine (`engine/mt5_interface.py`)**: The physical hand of the bot. It translates signal payloads into heavily validated, risk-adjusted MT5 order requests, managing everything from dynamic lot sizing to active trail-stops.

### 1.2 The "Two-Key" Verification System
To prevent the rapid over-trading common in high-frequency algorithmic systems, NEXUS employs a Two-Key logic gate for execution:
*   **Key 1 (Structural Validity):** The scanner maps market structure (Swing Highs/Lows) and identifies a "textbook" institutional footprint—such as a Fair Value Gap (FVG). If no structure exists, the engine sleeps.
*   **Key 2 (Momentum Validation):** Once an FVG is found, the Orchestrator passes a 9-dimensional snapshot of the current market (RSI, ADX, Distances from EMAs, Volatility) to the `RandomForestClassifier`. The AI outputs a probability score. Only if this score crosses the `AI_CONFIDENCE_THRESHOLD` (0.55) does the second key turn, initiating execution.

---

## 2. Mathematical Models & Risk Engines

### 2.1 Dynamic Position Sizing (The Risk Engine)
NEXUS calculates exact lot sizes dynamically at the millisecond of execution to enforce strict percentage-based risk parameters. 

**Formula:**
`Lots = (AccountBalance * (RiskPercentage / 100)) / (StopLossDistanceInPoints * TradeContractSize)`

*Example:* With a $100 balance, 1% risk, and a Stop Loss 25 points away on Gold (contract size 100):
`Lots = (100 * 0.01) / (25 * 100) = 1.0 / 2500 = 0.0004` (Rounded up to broker minimum `0.01`).

### 2.2 AI Confidence Scaling
If `DYNAMIC_RISK` is enabled, the bot scales the base risk up or down based on the AI's confidence score.
**Formula:**
`Scale = clamp(0.5, 1.5, ((AI_Conf - Threshold) / (0.95 - Threshold)) * 1.0 + 0.5)`
This ensures that if the AI is 95% confident, the bot risks 1.5x the normal amount, but if the AI barely clears the 55% threshold, it halves the risk.

### 2.3 Dynamic Risk-to-Reward (RR) Targeting
Take Profits (TP) are not hardcoded. They are mapped dynamically to structural Swing Highs/Lows. 
*   **Formula Check:** `Implied_RR = abs(TP_Price - Entry_Price) / abs(Entry_Price - SL_Price)`
*   If the structural target offers an `Implied_RR < 1.5`, the bot abandons the structure and enforces a strict mathematical `2.0R` target to preserve long-term mathematical expectancy.

### 2.4 Active Trade Management (The Defense System)
Once a trade is open, the engine actively protects unrealized profits:
*   **The Halfway Bank (Break-Even & Partial Close):** When price reaches 50% (`BE_TRIGGER_PCT = 0.5`) of the TP distance, the engine moves the SL to the exact Entry Price (Risk-Free) and closes 50% of the lot size to secure banked profit.
*   **The Shadow (Trailing Stop):** Once profit exceeds $4.50 (`TRAIL_START_DOLLARS`), the SL detaches and follows the current price exactly $3.00 (`TRAIL_DIST_DOLLARS`) behind it, securing profit against sudden "V-shape" market reversals.

---

## 3. Strategy Deep Dives

### 3.1 Smart Money Concepts (SMC) & FVG Mechanics
The primary strategy relies on identifying Fair Value Gaps (FVGs) on the M5 timeframe.
*   **Detection:** A 3-candle pattern where `Candle 3 Low > Candle 1 High` (Bullish FVG). The gap must exceed the `MIN_GAP_SIZE` (0.05).
*   **Invalidation & Inversion (iFVG):** If the price slices cleanly through an FVG, it invalidates the zone. The bot flips the zone into an **iFVG (Inverted FVG)**, anticipating that former support will act as resistance. 

### 3.2 The "Trap Inversion" Logic
NEXUS actively hunts retail traders getting trapped. If the M5 chart prints a beautiful Bullish FVG, but the H1 chart is in a massive downtrend (H1 ADX > 25, Price < H1 EMA50), the bot triggers a **Macro Trap Detection**. It deliberately ignores the M5 BUY signal and executes a **SELL** directly into the gap, front-running the macro institutional tide.

### 3.3 Confluence Modules
*   **London Breakout:** Automatically maps the Asian Session High/Low (1 AM to 7 AM UTC+2). Executes straddle limit orders on the breakout of these ranges during the volatile London open.
*   **VWAP Reversion:** Tracks standard deviations (Bands) from the Volume Weighted Average Price. Triggers mean-reversion trades when price pierces the 2nd deviation band and immediately curls back.

---

## 4. The "Full Might" Evolution Engine (Active Learning)

NEXUS is not static; it is a self-evolving entity. The Active Learning system operates in a continuous loop:

### 4.1 Forensic Data Logging (The Memory)
For every execution, `engine/mt5_interface.py` records a high-fidelity snapshot into `data/master_decision_log.csv`. This CSV contains:
*   **Execution Metadata:** Timestamp, Ticket, Signal Direction, Reason/Comment.
*   **Financial State:** Account Equity, Balance, and exact Broker Spread at the millisecond of entry.
*   **9-Dimensional Feature Vector:** `Dist_EMA_50`, `Dist_EMA_200`, `Dist_H1`, `RSI`, `RSI_Zone`, `Rel_Volatility`, `ADX`, `H1_RSI`, `H4_ADX`.

### 4.2 The "Brain Surgery" Phase (`evolve.py`)
After a batch of live trades, executing `python src/evolve.py` initiates the learning sequence:
1.  **Sync:** Connects to MT5 and pulls the exact historical PnL for the logged Tickets.
2.  **Diagnosis:** Evaluates *why* losses occurred (e.g., "Spread was > 100 points," "ADX was < 20").
3.  **Retraining:** Re-compiles the `RandomForestClassifier` using the new live data appended to the historical dataset, subtly shifting its internal decision trees to recognize and avoid the specific setups that resulted in losses.

### 4.3 Narrative Generation ("Letter to Father")
To provide transparency, the Evolution script utilizes Python formatting to generate a human-readable Markdown report in the `reports/` directory. It calculates current win rates, lists the primary failure vectors, and explicitly states how the neural weights were adjusted.

---

## 5. Historical Revisions & System Hardening

### 5.1 The "Goldilocks" Threshold (v2.1.0 Update)
Initially, the `AI_CONFIDENCE_THRESHOLD` was set to `0.60`. Live testing revealed that during aggressive crashes, the AI correctly identified the direction but yielded confidence scores of `0.56` to `0.59`, resulting in denied execution right before massive payouts.
*   **Resolution:** The threshold was lowered to `0.55`. This acts as a "Goldilocks" filter: it rejects pure coin-flips (`0.50 - 0.54`) but captures genuine momentum builds, vastly increasing execution frequency without sacrificing the integrity of the filter.

### 5.2 Critical Fixes
*   **CSV Parsing Alignment:** Feature vectors are now delimited via pipes (`|`) in the master log, preventing Pandas `ParserError` misalignments during comma-delimited feature unpacking.
*   **ValueError Unpacking Fix:** Resolved a critical bug in the `BUY FVG` block of `smc_fvg.py` where the engine expected 2 return values from the AI module but received 3 (Verdict, Confidence, Features), preventing total systemic crashes during bullish reversals.

---

## 6. Simulation & Backtesting Architecture

The `src/backtester.py` script provides a high-speed, vectorized simulation environment.
*   **Mechanics:** It uses `mt5.copy_rates_from_pos` to download thousands of historical M5 and H1 candles. It simulates the `goldvx.py` orchestrator loop by iterating through the Pandas arrays chronologically.
*   **Fidelity:** It explicitly simulates the 1-position limit, dynamic lot sizing, and evaluates the AI models against historical snapshots, providing a highly accurate representation of how the current model weights will perform over multi-month periods.

---
*End of Document. Generated automatically by the NEXUS Framework.*
