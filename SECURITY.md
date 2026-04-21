# Security Policy

## ⚠️ Financial Risk Disclaimer
**NEXUS is a highly experimental, high-frequency Quantitative Artificial Intelligence Engine. It is provided for educational and research purposes ONLY.** 

Algorithmic trading in leveraged financial markets (CFDs, Forex, Metals) carries a massive risk of capital loss.
* The authors and maintainers of NEXUS assume **zero liability** for any financial losses incurred from deploying this codebase in a live broker environment. 
* By running `src/goldvx.py`, you acknowledge that you are fully responsible for the sizing, limits, and killswitches inside `core/config.py`.

## 🛡️ Malware & Code Execution Warnings
Due to the nature of Python Machine Learning pipelines, this repository natively utilizes the `joblib` library to compress and load Neural Network matrices (`.joblib` files). 

* **NEVER download or inject third-party `.joblib` or `.pkl` files from unauthorized sources.** Native serialization pipelines in Python can execute malicious `exec()` or `os.system()` commands if unpickling a dynamically infected file.
* **Always build your own models natively** by strictly running `python src/auto_retrain.py` on your own secure hard-drive to dynamically compile the models from raw JSON price data. 

## 🚨 Reporting a Vulnerability

If you discover a structural security vulnerability (e.g., untrusted inputs, RCE vectors, API key leakage, or broker manipulation loopholes):

1. **Do NOT open a public GitHub issue.**
2. Privately email the repository maintainer directly.
3. Include explicit details on how the payload is triggered and which specific Python file holds the vector.

We will review the submission within 48 hours and coordinate a public patch once the core logic has securely neutralized the threat.
