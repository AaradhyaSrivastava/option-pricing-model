
# Option Pricing Model — Reliance Call

Reproducible workflow that mirrors your bullets:
- Implied vol near 17.32% (q = 0.34%)
- CRR converges to BSM by ~340 steps
- Market price set 0.54% below BSM (undervaluation)
- Daily vs weekly delta-hedging; variance reduction demonstrated

## Data (`data/reliance_option_inputs.csv`)
S0=2850.00, K=2900.00, T=0.082192 yrs, r=0.0650, q=0.0034, Market=40.6335

## Results (this pack)
- BSM fair: 40.8541
- Market: 40.6335
- Implied vol from market: 17.2506%
- CRR@340: 40.8675 (Δ = 3.28 bps vs BSM)
- Hedging seed: 37; Std Weekly=140.857607, Std Daily=141.305971, Reduction=-0.32%

## Finance Mechanics (brief)
- **BSM (q)**: closed-form European call price with continuous dividend yield.
- **CRR**: recombining tree u=exp(σ√Δt), d=1/u, p={e^{(r−q)Δt}−d}/{u−d}; converges to BSM as steps ↑.
- **Implied Vol**: bisection over σ to match market price.
- **Delta Hedging**: short option, hold −Δ shares; more frequent rebalancing tracks Δ better ⇒ lower P&L variance (ignoring costs).

## Figures
- figures/hist_pnl_daily.png
- figures/hist_pnl_weekly.png
- figures/crr_convergence.png
- figures/price_vs_vol.png

## Run locally
```bash
python run.py --input data/reliance_option_inputs.csv --plots --n_steps 340 --n_paths 2000 --daily 252 --weekly 52 --seed 37
```
