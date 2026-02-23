# LinkedIn Project Description

## Title
Bayesian Optimization for S&P 500 Pair Trading (Walk-Forward)

## Short Description (≤ 120 chars for profile card)
Walk-forward BO pair trading — 42 S&P 500 stocks, sector-constrained, annual re-selection, dollar-neutral, 2016–2025.

---

## Full Description

Built a **market-neutral quantitative pair trading system** applying Bayesian Optimization (Optuna TPE) with **quarterly walk-forward re-optimization and annual pair re-selection** — strategy parameters and pair universe both adapt to changing market regimes.

**What the project does:**
Screens 126 within-sector pairwise combinations across 42 S&P 500 stocks using dual cointegration testing (Engle-Granger + Johansen). Every quarter, Optuna runs 150 trials × 5-fold recency-weighted time-series CV to re-fit 4 parameters on the trailing 2 years. Every year, the cointegration screen re-runs and the top pairs are replaced if better candidates emerge — so the portfolio traded in 2024 (META/NVDA, OXY/SLB) differs from 2020 (CAT/GE, JNJ/LLY).

**Measured results (walk-forward OOS 2019–2025 vs fixed-parameter baseline):**
- META/NVDA: Sharpe −0.69 → **+0.24**; max drawdown **−78.5% → −10.1%**
- GS/WFC: Sharpe 0.47 → **0.74**; Calmar ratio **0.224 → 0.480**
- Walk-forward BO reduces max drawdown on **5/5 pairs** — stop-loss z-score working as designed

**Key engineering decisions:**
- Dollar-neutral return normalization (P&L / gross capital deployed) — avoids the spread pct-change instability of naive implementations
- Within-sector constraint on pair candidates — removes economically unjustified cross-sector pairings
- Parallel execution across pairs using joblib (4× speedup)
- Annual pair re-selection alongside quarterly parameter BO
- Adaptive portfolio: equal-weight top-2 pairs per year with full OOS equity curve vs SPY
- Attribution analysis: regime Sharpe heatmap, trade frequency, pairwise return correlation

**Stack:** Python · Optuna · statsmodels · yfinance · Plotly · pandas · NumPy · joblib

---

## Skills Tags
Python · Quantitative Finance · Algorithmic Trading · Bayesian Optimization · Statistical Modeling · Time Series Analysis · Machine Learning

---

## Resume Bullet Points (pick one)

**Option A — technical/data science roles:**
> Built walk-forward Bayesian Optimization pipeline (Optuna TPE, 150 trials, 5-fold time-series CV) for market-neutral S&P 500 pair trading; annual pair re-selection via dual cointegration testing (EG + Johansen) on 126 within-sector candidates; dollar-neutral return normalization; parallel execution (joblib); walk-forward BO reduced max drawdown from −78.5% to −10.1% on META/NVDA and improved Calmar ratio 2× on GS/WFC across 4 market regimes (2016–2025).

**Option B — shorter, quant/finance roles:**
> Engineered adaptive pair trading system with quarterly walk-forward BO (Optuna) and annual pair re-selection; within-sector cointegration constraint, dollar-neutral positioning, stop-loss z-score as 4th parameter; BO improved Sharpe on 4/5 pairs and reduced max drawdown by 50–90% vs fixed baseline across COVID crash, rate hike cycle, and AI rally.

**Option C — general analyst roles:**
> Developed pair trading backtest with Bayesian Optimization (Optuna) and walk-forward re-optimization across 42 S&P 500 stocks; 4 optimized parameters including stop-loss; annual pair re-selection; Plotly attribution dashboard; drawdown reduced 50–90% over fixed-parameter baseline.

---

## How to Frame in an Interview

If asked "your strategy underperforms SPY" — say:
> "Pair trading is market-neutral by design — I'm simultaneously long one stock and short another, so market beta is near zero. Comparing it to a directional long-only SPY return during the 2019–2025 bull market isn't the right benchmark. The correct question is: does it generate positive risk-adjusted returns relative to cash (0%), and does the walk-forward BO improve over the fixed baseline? On the first: GS/WFC delivered +30% with Sharpe 0.74. On the second: BO improved Sharpe on 4/5 pairs and reduced max drawdown on all 5."

If asked "why are win rates only 3–5%?" — say:
> "That's time *in position*, not trade win rate. The BO finds conservative entry thresholds (high entry_z) to avoid transaction cost drag on short quarterly windows. It's trading infrequently but efficiently — which is appropriate for a mean-reversion strategy where being out of the market is a valid position."

---

## LinkedIn Post (for when you publish the GitHub repo)

Just finished a project I'm proud of — sharing both the methodology and honest results.

📈 **Adaptive Walk-Forward Bayesian Optimization for S&P 500 Pair Trading**

The main insight: most BO backtests fit parameters once on all historical data. That's look-ahead bias in disguise. This project re-optimizes every quarter using only the trailing 2 years — and re-runs the pair selection screen every year so the strategy adapts to which sectors are actually cointegrated right now.

Notable results (market-neutral OOS, 2019–2025):
✅ META/NVDA: max drawdown −78.5% → −10.1% (stop-loss z-score working)
✅ GS/WFC: Calmar ratio doubled (0.22 → 0.48)
✅ Sharpe improved on 4/5 pairs vs fixed-parameter baseline
📊 Annual pair selection log shows regime shifts: energy pairs dominated 2021–2022, tech pairs in 2024

Engineering highlights: dollar-neutral returns, within-sector cointegration constraint, parallel joblib execution, 6-panel Plotly dashboards.

GitHub: [link]

#QuantitativeFinance #Python #BayesianOptimization #AlgorithmicTrading #DataScience
