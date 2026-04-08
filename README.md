# 🚀 NEXUS Quantitative Trading Matrix

[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![MetaTrader 5](https://img.shields.io/badge/MetaTrader-5-green.svg)](https://www.metatrader5.com/)
[![Scikit-Learn](https://img.shields.io/badge/AI-Scikit--Learn-orange.svg)](https://scikit-learn.org/)

NEXUS is an autonomous, institutional-grade, multi-threaded Artificial Intelligence trading orchestrator. It is built strictly for the `XAUUSD` (Gold) exchange on MetaTrader 5 but is mathematically agnostic. Instead of relying on a single stagnant trading condition, NEXUS functions as a "Master Orchestrator," dynamically routing real-time market data across a suite of 5 distinctly independent quantitative engines. These engines natively deploy, throttle, and sleep based entirely on macroscopic market regimes (Trending, Choppy, Dead, or Squeezed).

Furthermore, NEXUS integrates a dual-layered local **Machine Learning Gatekeeper**. Every trade mapped by the geometry engines must pass a rigorous multidimensional matrix check against a `.joblib` Random Forest Classifier before physical capital is ever risked.

---

## 🏛️ Orchestrator Design & Market Regimes

The `goldvx.py` core daemon serves as the central circulatory system. It requests raw tick-level telemetry from MetaTrader 5, filters out violent macro-economic news environments (FOMC, CPI, NFP) via independent API routing, and establishes the "Regime" of the market to decide which analytical weapons to unholster.

NEXUS uses `ADX (Average Directional Index)` and Volatility multi-dimension filters:
* **[HOT / TREND] Mode (`ADX > 25`)**: The market has a clean, undisputed vector. The Orchestrator disables mean-reverting algorithms and turns on the **SMC FVG** and **SMC Order Block** mapping algorithms. 
* **[RANGE] Mode (`ADX < 20`)**: The market is trapped in institutional chop. Breakout and structural tracking are disabled. The **RSI AI Scalping** and **VWAP Reversion** models activate to physically fade the extreme deviations.
* **[FLAT] Mode**: During dangerous cross-sections like the NY/London Overlap phase, NEXUS entirely disables new trade deployments but continues actively managing stop-losses and trailing stops on currently floating trades to protect capital from erratic spread conditions.

---

## 🧠 The 5 Quantitative Matrix Engines

NEXUS contains five separate, purely mathematical processing scripts located in the `/strategies` framework. Rather than fighting each other, they operate congruently on the same tick data stream:

### 1. Advanced Structural SMC Order Blocks (`strategies/smc_orderblock.py`)
This engine utilizes native recursive fractaling to map historical "Breaks of Structure" (BOS). 
When a confirmed BOS is painted indicating a shift in momentum, the algorithm rewinds time to find the *absolute last opposing institutional candle* before the impulse. It boxes this candle (wick-to-wick) as the true Order Block. If price mathematically drifts back into this isolated block, it executes with Stop-Losses tethered precisely `0.2` Standard Deviations below the structural base wick to prevent broker spread-hunting.

### 2. Fair Value Gap Consolidation (`strategies/smc_fvg.py`)
This volumetric inefficiency engine scans 3 consecutive M5 candles to calculate empty price spaces (Gaps) where extreme institutional slippage occurred. These FVGs act as magnetic liquidity traps. However, NEXUS enforces an internal "Macro Trend Guard." It will deliberately invert trades if an M5 FVG is detected moving opposing the H1/H4 primary exponential moving average (EMA) currents.

### 3. Mean Volatility Breakout (`strategies/bb_breakout.py`)
Deployed strictly when the market goes dead. This engine calculates absolute BandWidth (The deviation quotient between Upper and Lower Bollinger Bands). When the kinetic energy drops to the lowest `15%` recorded across the last 100 intervals, the `[SQUEEZE_ACTIVE]` flag is thrown. The instant a candle ruptures OUTSIDE the squeezed bands, accompanied by a Tick Volume multiple of `>= 2.0x` indicating explosive kinetic release, NEXUS rides the wave.

### 4. High-Frequency VWAP Stretch (`strategies/vwap_reversion.py`)
Identical logic to HFT Mean Reversion desks. The VWAP (Volume Weighted Average Price) anchors the intrinsic "Fair Value" for the 24-hour cycle using tick-volume clustering. As price strays, NEXUS measures the deviation mathematically. If price abruptly rockets `> 3.0` standard deviations away from the VWAP anchor due to low-liquidity spikes, NEXUS triggers an aggressive counter-reversal to short the stretched rubber band. 

### 5. RSI/AI Oscillation Scraper (`strategies/rsi_reversion.py`)
A pure chop-survival tool. It relies on the raw, hyper-local momentum indicators. Instead of relying purely on mathematically crossing "overbought" indices (e.g. `RSI > 70`), it bundles `RSI`, `ROC` (Rate of Change), and `Rel_Volatility` parameters into a Matrix to determine exactly when a swing high has fully exhausted its internal buyer momentum.

---

## 🦾 Artificial Intelligence Protocol & Joblib Matrix

Algorithms are mathematically blind. They cannot 'feel' the market condition. Because of this, NEXUS completely strips the execution authority out of the algorithms and passes it to the AI matrix.

Inside `train_model.py`, millions of historical rows of XAUUSD tick data have been harvested. 16 defining characteristics (such as the Distance to the 200 EMA, Relative Candle Size, RSI 2-period trajectory delays, and 5-period Momentum Velocity) were dumped into a massive Scikit-Learn **Random Forest Classifier** which generated two localized brains:
1. `gold_trend_model.joblib`: Trained to evaluate setups moving *with* the macro trend.
2. `gold_reversal_model.joblib`: Trained to evaluate setups attempting to snipe the peaks of chaotic ranging.

### The Judgment Loop
When `SMC_FVG` draws a zone and wants to risk money, it is forced to submit a live `X_live` matrix of the current 16 features to the AI.
The AI runs `.predict_proba()` against its 300 pre-trained decision trees. It returns a deterministic confidence score (e.g., `0.48` chance of success). 
Since the `config.py` enforces a `0.55` hard confidence barrier, the trade is instantly labeled `[AI DENIED]` and your capital is securely saved from poor mechanics.

---

## 🛠️ Installation & Virtual Setup

To safely run the deployment inside an isolated cluster, open your terminal:

```bash
# 1. Clone the repository
git clone https://github.com/Ugochi56/Nexus-Trading-Bot.git
cd Nexus-Trading-Bot

# 2. Setup your dependencies
pip install -r requirements.txt
```

> [!CAUTION]  
> **MISSING INTELLIGENCE FILES**
> To prevent leaking proprietary quantitative models, the AI Brains (`*.joblib` files) have been strictly excluded from this repository via `.gitignore`. 
> **You MUST manually acquire and inject `gold_trend_model.joblib` and `gold_reversal_model.joblib` into the root directory before running the bot!** Otherwise, the NEXUS system will trigger a critical emergency shutdown.

---

## ⚙️ Risk Management & Terminal Parameters

Inside `core/config.py`, all mechanical constraints are rigidly coded. 

#### Equity Defense System (Kill-Switches)
* `MAX_DAILY_LOSS_PERCENT = 3.0`: The engine calculates your exact capital at 00:00 server time. If the PnL ever drifts globally to `-3.0%`, the Orchestrator executes a hard `[HALT]` routine and will not execute further until the next calendar day.
* `NEWS_KILL_SWITCH = True`: Connects natively to the ForexFactory backend JSON Database, circumventing unreliable HTML scraping. If an underlying `High-Impact` Event is localized, NEXUS pauses operations exactly 60 minutes prior and resumes execution 60 minutes after.
* `CLOSE_ALL_ON_FRIDAY = True`: Strictly halts weekend gap-risk by physically liquidating all open positions on the MT5 books at 21:00 UTC Friday.

#### Trailing Physics
* `BE_TRIGGER_PCT = 0.5`: The exact fraction a trade must travel toward Take Profit before scaling 50% lotsize and jamming the Stop-Loss purely to Break Even.
* `USE_TRAILING_STOP = True`: Natively trails mathematically using a 14-period True Range calculation. No generic pips are used, it adapts natively to the volatility curve of the asset.

---

## 🚀 Initiating Core Sequence

Deploying the Engine is a single command. The Python thread binds implicitly to the highest active user-privilege MetaTrader 5 terminal process on your machine.

```bash
python goldvx.py
```

The terminal will physically collapse the noise and present a fully professional high-frequency string tracking exactly what the internal state architecture is executing:
```text
[START] Booting NEXUS Core...
[SUCCESS] Connected: 3866959 | $13,068.87
=======================================================
[NEXUS ONLINE]
[ENGINES: SMC_FVG, SMC_OB, RSI, VWAP, BB_BREAKOUT]
=======================================================
[DAY] New Day Equity: $13208.87
[CACHE] Loaded ForexFactory Calendar from local cache.

[CHANGE] Switched state to [TREND]. Active Engines: SMC_FVG | SMC_OB | BB_BREAKOUT
[LDN[DST]] [HOT] Trend:UP | ADX:39.4 | Bal: $13,208.87 | Prc: $4798.41
```
