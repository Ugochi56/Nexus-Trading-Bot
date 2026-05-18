# NEXUS Core: The Quantitative Engineering Manual
## Advanced Algorithmic Trading & Active Machine Learning
**Version:** 2.1.0 ("Full Might" Evolution)
**Generated:** May 18, 2026

---

## Table of Contents
1. [Executive Summary & Philosophy](#1-executive-summary--philosophy)
2. [Quantitative Foundations](#2-quantitative-foundations)
    - 2.1 Smart Money Concepts (SMC) Theory
    - 2.2 Random Forest Machine Learning Integration
3. [System Architecture & Topography](#3-system-architecture--topography)
    - 3.1 The Orchestrator Node (`goldvx.py`)
    - 3.2 The Logic Hub (`strategies/`)
    - 3.3 The Execution Bridge (`mt5_interface.py`)
4. [Mathematical Risk Engines](#4-mathematical-risk-engines)
    - 4.1 Dynamic Position Sizing Algorithms
    - 4.2 AI Confidence Scaling Multipliers
    - 4.3 Structural Risk-to-Reward Targeting
    - 4.4 Trade Defense Mechanisms (BE & Trailing)
5. [Strategy Deep Dives](#5-strategy-deep-dives)
    - 5.1 Fair Value Gaps (FVG) & Trap Inversions
    - 5.2 Confluence Modules (VWAP, BB, Liquidity)
6. [The "Full Might" Evolution Engine](#6-the-full-might-evolution-engine)
    - 6.1 Forensic Data Logging (`master_decision_log.csv`)
    - 6.2 Autonomous Retraining Protocols (`evolve.py`)
7. [Simulation & Backtesting Architecture](#7-simulation--backtesting-architecture)
8. [Deployment & Maintenance](#8-deployment--maintenance)

---



## 1. Executive Summary & Philosophy

The NEXUS Trading Bot is not a simple heuristic script. It is a highly advanced, multi-threaded quantitative trading engine specifically engineered for the high-volatility XAUUSDm (Gold) CFD market. 

At its core, NEXUS operates on a hybrid philosophy: **Structural Scanning paired with Algorithmic Momentum Validation.** Human traders often fail because they lack the discipline to filter out sub-optimal structural setups. Traditional algorithmic bots fail because they rely purely on lagging indicators (like moving averages) without understanding the physical price-action structure of the market.

NEXUS solves this by deploying a "Two-Key" verification system. Key 1 is purely structural—identifying institutional footprints like Fair Value Gaps (FVGs) and Order Blocks. Key 2 is purely mathematical—feeding a 9-dimensional vector of current market conditions into a trained `RandomForestClassifier` to determine if the physical structure is supported by underlying momentum.

With the recent introduction of the **"Full Might" Evolution Engine**, NEXUS has transitioned from a static execution script into an Active Learning entity. It records its own failures, parses broker conditions at the millisecond of entry, and autonomously retrains its neural weights to adapt to shifting market regimes.



## 2. Quantitative Foundations

### 2.1 Smart Money Concepts (SMC) Theory
Traditional retail trading relies on support/resistance and trendlines. NEXUS utilizes Smart Money Concepts (SMC), which hypothesizes that markets are moved by large institutional orders that leave distinct structural footprints.

**The Fair Value Gap (FVG)**
An FVG occurs when institutional volume enters the market so aggressively that it leaves a "gap" in price action where no trading occurred between buyers and sellers. Mathematically, NEXUS defines a Bullish FVG as a 3-candle sequence where the low of Candle 3 is strictly greater than the high of Candle 1 by a minimum delta (`MIN_GAP_SIZE`).
The theory states that price will eventually retrace to "fill" this gap to balance the market. NEXUS places execution zones precisely at the borders of these gaps.

**Inversions & Traps**
When an FVG is violently broken, it implies the institutional momentum was a trap or has fundamentally shifted. NEXUS immediately re-classifies the broken zone as an **Inverted FVG (iFVG)**, flipping its directional bias to ride the new momentum.

### 2.2 Random Forest Machine Learning Integration
To validate these structural setups, NEXUS utilizes `scikit-learn`'s `RandomForestClassifier`. 
A Random Forest is an ensemble learning method that constructs a multitude of decision trees at training time.

**The Feature Matrix**
For every setup, NEXUS calculates a 9-dimensional snapshot:
1. `Dist_EMA_50`: The percentage distance from the micro-trend.
2. `Dist_EMA_200`: The percentage distance from the macro-trend.
3. `Dist_H1`: The percentage distance from the 1-Hour institutional trend.
4. `RSI` & `RSI_Zone`: Relative Strength Index metrics to gauge exhaustion.
5. `Rel_Volatility`: The current candle size relative to the Average True Range (ATR).
6. `ADX`: The Average Directional Index (Trend Strength).
7. `H1_RSI` & `H4_ADX`: Macro-level momentum filters.

The algorithm splits these features across hundreds of decision trees, calculating the Gini impurity at each node to determine the optimal classification boundary. The final output is not a binary "Yes/No", but a raw probability distribution (e.g., 58% SELL, 42% BUY).



## 3. System Architecture & Topography

The codebase is heavily modularized to ensure thread-safety, speed, and logical separation of concerns.

### 3.1 The Orchestrator Node (`src/goldvx.py`)
This is the infinite execution loop. Its primary duties include:
- **State Synchronization:** Ensuring the MT5 terminal is connected and equity trackers are accurate.
- **Data Polling:** Fetching the latest `M5`, `H1`, and `H4` Pandas DataFrames.
- **Strategy Multiplexing:** Iterating through the active strategies (`SMC_FVG`, `VWAP`, `BB_BREAKOUT`, etc.), passing the dataframes, and awaiting signal payloads.
- **Event Routing:** If a signal payload is received and the engine is not actively in a trade, it routes the payload to the Execution Bridge.

### 3.2 The Logic Hub (`strategies/`)
The `strategies` directory acts as a pure mathematical sandbox. Scripts in this folder (like `smc_fvg.py`) have absolutely no knowledge of the MT5 broker, account balance, or open trades. They receive DataFrames, perform vectorized Pandas operations, consult the AI models, and return a standardized dictionary payload containing:
- `signal`: 'BUY' or 'SELL'
- `sl`: The structural Stop Loss price.
- `tp_price`: The structural Take Profit price.
- `confidence`: The AI probability score.
- `features`: The raw 9-dimensional array used for the prediction.

### 3.3 The Execution Bridge (`engine/mt5_interface.py`)
This module is the physical hand of the bot. It translates the abstract payloads into strict C-level MT5 `order_send` requests. It handles slippage (Deviation), magic number tagging, execution logging, and active trade management (Trailing Stops).



## 4. Mathematical Risk Engines

NEXUS does not use static lot sizes. Every single execution is calculated dynamically to ensure exact risk distribution regardless of how wide or tight the Stop Loss is.

### 4.1 Dynamic Position Sizing Algorithms
When `mt5_interface.py` receives an execution signal, it fires the `calculate_position_size` function.

```python
sl_dist = abs(entry_price - sl_price)
lots = (account.balance * (risk_pct / 100)) / (sl_dist * symbol_info.trade_contract_size)
```
If the account balance is $100 and the risk is 1%, the bot intends to lose exactly $1.00 if the Stop Loss is hit. If the FVG is massive (a 50-point SL), the bot will calculate a microscopic lot size. If the FVG is incredibly tight (a 10-point SL), it will calculate a much larger lot size. This ensures the geometric compounding of the account remains perfectly smooth.

### 4.2 AI Confidence Scaling Multipliers
The baseline risk (e.g., 1%) is further modified by the AI's confidence.
If `DYNAMIC_RISK` is enabled:
```python
scale = max(0.5, min(1.5, (ai_conf - AI_CONFIDENCE_THRESHOLD) / (0.95 - AI_CONFIDENCE_THRESHOLD) * 1.0 + 0.5))
risk_pct = risk_pct * scale
```
If the AI confidence perfectly matches the threshold (0.55), the scale is 0.5x (cutting risk in half for uncertain trades). If the AI is 95% confident, the scale is 1.5x (betting heavily on high-probability setups).

### 4.3 Structural Risk-to-Reward Targeting
Rather than using a fixed math-based TP (like exactly 2.0x the SL), NEXUS maps the market structure using `map_market_structure()` to find the nearest Swing High or Swing Low. It places the Take Profit exactly at that liquidity pool.
However, it enforces a mathematical floor: `MIN_DYNAMIC_RR = 1.5`. If the nearest Swing High is too close and only offers a 1.1 RR, the bot rejects the structural TP and enforces a strict mathematical `RISK_REWARD_RATIO = 2.0` to preserve the system's edge.

### 4.4 Trade Defense Mechanisms
NEXUS employs a strict capital preservation routine inside `manage_open_positions()`:
1.  **The 50% Break-Even Trigger:** If the trade travels 50% of the distance to the TP, the engine immediately modifies the order, moving the Stop Loss to the Entry Price, and liquidating 50% of the position to secure banked profit.
2.  **The Shadow Trail:** Once a trade is deeply in profit (>$4.50), the SL detaches and follows the current price at a fixed distance ($3.00), locking in gains against sudden "V-Shape" reversals.



## 5. Strategy Deep Dives

### 5.1 Fair Value Gaps (FVG) & Trap Inversions
The flagship strategy (`smc_fvg.py`) is a masterpiece of quantitative trap-hunting.
- **FVG Detection:** Uses a rolling 30-candle window to identify `c3['low'] > c1['high']`.
- **The "Two-Key" Gate:** As discussed, it requires both the structural FVG and an AI `prob > 0.55`.
- **Trap Front-Running:** If the AI predicts `SELL`, but the structure is a `BUY` FVG, the bot checks the H1 Macro Confluence. If the Macro trend is heavily bearish, the bot recognizes the `BUY` FVG as a "Retail Trap." Instead of ignoring the trade, it dynamically inverts the payload, executing a `SELL` directly into the gap to crush the trapped retail buyers.

### 5.2 Secondary Confluence Modules
To ensure the bot operates in all market environments, secondary engines are engaged:
- **VWAP Reversion (`vwap_reversion.py`):** Calculates the Volume Weighted Average Price with Standard Deviation bands. If price extends beyond the 2nd deviation (extreme extension) and curls back inward, it executes a mean-reversion trade toward the central VWAP line.
- **RSI Reversion (`rsi_reversion.py`):** Hunts for severe exhaustion (RSI > 80 or < 20). If combined with a reversal AI model prediction, it snipes the exact top or bottom of a trend.
- **London Breakout (`london_breakout.py`):** Calculates the high and low of the low-volume Asian Session. During the London open (highest volume period), it places straddle pending orders, anticipating a massive liquidity expansion in one direction.



## 6. The "Full Might" Evolution Engine

In Version 2.1.0, NEXUS was upgraded from a static script into an Active Learning neural network.

### 6.1 Forensic Data Logging
Every execution triggers the Forensic Logger inside `mt5_interface.py`. It writes a detailed row to `data/master_decision_log.csv`.
Crucially, this is not just a list of trades. It is the exact **"State of Mind"** of the bot at the millisecond of execution. It records the Broker Spread, the Account Equity, the AI's exact confidence score, and the 9-dimensional Feature Vector.
*Technical Note:* To prevent Pandas parsing errors, the feature vectors are delimited by pipes (`|`) within the CSV to ensure column alignment.

### 6.2 Autonomous Retraining Protocols (`evolve.py`)
After a sample size of live trades, the `evolve.py` script is executed. 
1.  **Reconciliation:** It connects to MT5 and pulls the exact historical deal history, matching order tickets to the internal `master_decision_log.csv`.
2.  **Profit/Loss Tagging:** It updates the local dataset, marking each feature vector as a `1` (Win) or `0` (Loss).
3.  **Neural Reweighting:** The script isolates the failure vectors (e.g., trades lost due to extremely low ADX or abnormally high broker spread) and retrains the `RandomForestClassifier`. The AI essentially "learns" that taking trades when the Spread is > 50 points results in a 0% win rate, and will mathematically down-weight those setups in the future.

### 6.3 Narrative Generation
To keep the human operator informed, `evolve.py` utilizes string formatting to generate a `Markdown` report ("Letter to Father") detailing exactly *why* it failed, what metrics caused the failure, and how it has adjusted its internal weights.



## 7. Simulation & Backtesting Architecture

NEXUS includes a custom-built, hyper-fast vectorized backtester (`src/backtester.py`).

### 7.1 Vectorized Execution Simulation
Standard backtesters (like MT5 Strategy Tester) iterate tick-by-tick, which takes hours. The NEXUS backtester downloads the raw OHLC data from the broker into Pandas DataFrames and simulates months of trading in seconds.

**Mechanics:**
It slices the arrays (`df.iloc[i-200 : i+1]`) to recreate the exact view the bot would have had at that historical moment. It passes this slice to the exact same `evaluate()` functions used in live trading, ensuring 100% parity between backtest logic and live logic.

### 7.2 The `0.55` "Goldilocks" Threshold Optimization
Extensive live logging and backtesting revealed a critical optimization flaw. At `0.60` confidence, the bot was denying highly profitable trades because the AI was returning `0.58` and `0.59` scores during massive, sudden market crashes. 
The configuration was permanently altered to `0.55`. This filter still successfully rejects pure 50/50 coin-flips but allows the bot to capture the massive momentum explosions that previously fell just short of the strict 60% requirement.



## 8. Deployment & Maintenance

### 8.1 Server Operations
NEXUS is designed to run on a Virtual Private Server (VPS) with sub-10ms latency to the broker's trading servers. 

### 8.2 Maintenance Protocols
- **Weekly Restarts:** Ensure memory leaks within the `MetaTrader5` Python C-library are flushed.
- **Model Evolution:** The `evolve.py` script should be run every 20-50 trades to ensure the AI weights are continuously adapting to the current market regime. 
- **Log Cleaning:** The `master_decision_log.csv` will grow indefinitely. While this provides more training data, it should be periodically backed up to prevent file corruption.

---
**NEXUS Core Engineering Division**
*End of Technical Manual.*
