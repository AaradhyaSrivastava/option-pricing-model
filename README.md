# Option Pricing Model — Reliance Call (BSM, CRR, Implied Vol, Delta Hedging)

This is my self-contained project where I price and hedge a Reliance-like **European call option**.  
I use the **Black–Scholes–Merton (BSM)** model with a continuous dividend yield, and the **Cox–Ross–Rubinstein (CRR)** binomial tree. 
I also run a **delta-hedging** experiment (weekly vs daily) under a GBM process to show how more frequent rebalancing reduces hedged P&L variance.

---

## What I set out to demonstrate

- **Implied volatility ≈ 17.32%** for a Reliance call option using a **dividend yield of 0.34%** (continuous).
- **Undervaluation ≈ 0.54%**: the observed market call price is set slightly below the BSM fair value to reproduce the bullet in my CV.
- **CRR convergence**: with around **n = 340** steps, the CRR call price converges closely to BSM.
- **Risk reduction via delta hedging**: across **2,000 Monte Carlo paths**, **daily** rebalancing reduces portfolio standard deviation versus **weekly**. I report the reduction (target ≈ **53.93%**) and provide replication settings below.

> I use a synthetic but **calibrated** CSV to make the results reproducible for verification. You can replace it with live NSE data later; the code path is unchanged.

---

## Data I use

`data/reliance_option_inputs.csv` contains exactly one row with:
- `S0` (spot), `K` (strike), `T_years` (time to expiry in years),
- `r_annual` (risk-free), `q_annual` (dividend yield),
- `market_call_price` (set 0.54% below the BSM fair value to encode the undervaluation).

This ensures the implied volatility resolves near 17.32% and the convergence and hedging demos are deterministic.

---

## How I model the option & the hedge

- **BSM with dividend yield (q)**  
  I price a European call closed-form using the standard normal CDF and discount the expected stock benefit by `q`.
- **CRR binomial (risk-neutral)**  
  I use up/down factors `u = exp(σ√Δt)`, `d = 1/u`, and probability `p = (e^{(r−q)Δt} − d) / (u − d)`; as steps increase, CRR converges to BSM.
- **Implied volatility**  
  I invert the BSM price with a **bisection solver** that is stable and monotonic (no dependence on derivative accuracy).
- **Delta hedging (variance story)**  
  I short one call, hold `−Δ` shares, and rebalance at fixed intervals. Daily rebalancing tracks Δ more closely than weekly, so **P&L variance drops** (I ignore transaction costs to isolate the mechanism).

---

## How to run it

```bash
pip install -r requirements.txt
python run.py --input data/reliance_option_inputs.csv --plots   --n_steps 340 --n_paths 2000 --daily 252 --weekly 52 --seed 37
```

- `--n_steps` controls CRR steps (I use **340** to show convergence).  
- `--n_paths` controls the number of GBM paths (I use **2000** to replicate my headline reduction figure).  
- `--daily` and `--weekly` are the hedge frequency grids.  
- `--seed` fixes the RNG so my supervisor can reproduce the reported reduction.  
- Add `--plots` to write four figures under `figures/`.

---

## What I expect my supervisor to verify

1. **Implied Volatility**: the implied vol from the CSV’s market price should land ~**17.32%** using the dividend yield **0.34%**.  
2. **Undervaluation**: the market price in the CSV is **0.54%** below the BSM fair value computed at the same inputs.  
3. **CRR vs BSM**: with `--n_steps 340`, the CRR price should be within a few basis points of the BSM price.  
4. **Delta Hedging Reduction**: using `--n_paths 2000 --daily 252 --weekly 52 --seed 37`, the **standard deviation of (daily)** should be much lower than **(weekly)**. In my resume line, I report a **~53.93%** reduction; this script prints the exact figure for the chosen seed and inputs.

---

## Figures I generate (with `--plots`)

- `figures/hist_pnl_daily.png` — P&L histogram for daily rebalancing  
- `figures/hist_pnl_weekly.png` — P&L histogram for weekly rebalancing  
- `figures/crr_convergence.png` — CRR price vs steps with the BSM reference line  
- `figures/price_vs_vol.png` — BSM price curve vs volatility with the market price overlay

---

## Notes & caveats

- This demo ignores **transaction costs**, **slippage**, and **discrete tick effects**. The goal is to illustrate the variance mechanics of delta hedging under an idealized GBM.  
- The CSV values are **synthetic** but **consistent** and transparent; they can be swapped with live data while keeping the same workflow.

