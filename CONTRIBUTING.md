# Contributing to NEXUS Trading Bot

First off, thank you for considering contributing to NEXUS! It's people like you that make NEXUS the powerful quantitative machine that it is.

## 🧠 Architectural Philosophy
NEXUS is designed as a **Defense-First Quantitative AI Engine**. Every pull request should adhere to the following core tenets:
1. **Mathematical Edge Required:** We do not accept strategies based on generic retail logic (e.g., standard moving average crossovers). All PRs containing strategy engines must have a provable statistical edge (e.g., specific volatility thresholds, VWAP standard-deviation stretches, etc.).
2. **Defensive Hardening:** Any logic added must cleanly route through the `engine/mt5_interface.py` killswitches. No side-loading trades around the Daily Drawdown limits. 

## 🛠️ How to Contribute

### 1. Bug Reports & Feature Requests
Please use the GitHub Issue Tracker to report bugs or request features. When submitting:
* Clearly describe the issue.
* Provide MT5 logs showcasing the execution bug.
* Mention the specific broker and account type (Standard vs Micro).

### 2. Pull Requests
1. **Fork the repository** and create your branch from `main`.
2. **Write Clean Pythonic Code:** Ensure your code adheres to standard PEP8 structural formatting.
3. **Machine Learning Updates:** If you are modifying the ML feature logic, you **must** update `src/auto_retrain.py` to ensure the Dual-Brain Orchestrator correctly evaluates your new features. *Do not push your local `.joblib` files to the repository.* Let users explicitly retrain the models locally.
4. **Issue a PR:** Detail exactly what your mathematical theory is and what edge the code exploits.

### 3. Local Model Retraining
If you are contributing to the Scikit-Learn pipelines, note that `gold_trend_model.joblib` and `gold_reversal_model.joblib` are explicitly stripped from git tracking for security. When testing your branch, ensure you run `python src/auto_retrain.py` locally to build your AI brains before testing the `goldvx.py` engine.
