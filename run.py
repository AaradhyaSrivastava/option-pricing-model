import math
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
import matplotlib.pyplot as plt
from dataclasses import dataclass

# ------------------ Black–Scholes ------------------
def bsm_call_price(S, K, T, r, sigma, q=0.0):
    if T <= 0 or sigma <= 0:
        return max(0.0, S*math.exp(-q*T) - K*math.exp(-r*T))
    d1 = (math.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*math.sqrt(T))
    d2 = d1 - sigma*math.sqrt(T)
    return S*math.exp(-q*T)*norm.cdf(d1) - K*math.exp(-r*T)*norm.cdf(d2)

def bsm_call_delta(S, K, T, r, sigma, q=0.0):
    if T <= 0 or sigma <= 0:
        return 1.0 if S > K else 0.0
    d1 = (math.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*math.sqrt(T))
    return math.exp(-q*T)*norm.cdf(d1)

def implied_vol_call_bsm(S, K, T, r, q, market_price):
    f = lambda sig: bsm_call_price(S, K, T, r, sig, q) - market_price
    return brentq(f, 1e-6, 3.0)

# ------------------ CRR Binomial ------------------
def crr_binomial_call(S, K, T, r, sigma, q=0.0, steps=340):
    dt = T/steps
    u  = math.exp(sigma*math.sqrt(dt))
    d  = 1.0/u
    a  = math.exp((r - q)*dt)
    p  = (a - d)/(u - d)
    j = np.arange(steps, -1, -1)
    ST = S * (u**j) * (d**(steps-j))
    payoff = np.maximum(ST - K, 0.0)
    disc = math.exp(-r*dt)
    for _ in range(steps):
        payoff = disc*(p*payoff[:-1] + (1-p)*payoff[1:])
    return float(payoff[0])

# ------------------ GBM Simulation ------------------
@dataclass
class HedgeResult:
    mean: float
    std: float
    pnl: np.ndarray

def simulate_paths_gbm(S0, mu, sigma, T, n_steps, n_sims, seed=81):
    rng = np.random.default_rng(seed)
    dt = T/n_steps
    Z  = rng.standard_normal((n_sims, n_steps))
    increments = (mu - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z
    S = np.empty((n_sims, n_steps+1))
    S[:,0] = S0
    S[:,1:] = S0*np.exp(np.cumsum(increments, axis=1))
    return S

def buyers_hedge_pnl(S_paths, K, r, q, sigma_iv, T, rebalance='daily'):
    n_sims, n_cols = S_paths.shape
    dt = T/(n_cols-1)
    if rebalance == 'weekly':
        mask = np.zeros(n_cols, dtype=bool); mask[::5] = True; mask[-1] = True
    else:
        mask = np.ones(n_cols, dtype=bool)
    pnls = np.empty(n_sims)
    for i in range(n_sims):
        S = S_paths[i]
        cash = 0.0
        C0 = bsm_call_price(S[0], K, T, r, sigma_iv, q)
        delta0 = bsm_call_delta(S[0], K, T, r, sigma_iv, q)
        shares = -delta0
        cash  += -C0 + delta0*S[0]
        for t in range(1, n_cols):
            cash *= math.exp(r*dt)
            tau = max(T - t*dt, 0.0)
            if mask[t]:
                new_delta = bsm_call_delta(S[t], K, tau, r, sigma_iv, q)
                d_sh = -new_delta - shares
                cash += - d_sh * S[t]
                shares += d_sh
        payoff = max(S[-1]-K,0.0)
        cash += shares*S[-1]
        cash += payoff
        pnls[i] = cash
    return HedgeResult(mean=float(pnls.mean()), std=float(pnls.std()), pnl=pnls)

# ------------------ Inputs ------------------
S0 = 1375.0
K  = 1400.0
r_cc = math.log(1+0.058)     # risk-free (continuous)
q_cc = math.log(1+0.0034)    # dividend yield (continuous)
T1 = 30/365                  # 1-month contract

# ===================================================
# Part 1 — Implied Volatility from market premium
# ===================================================
print("\n=== Part 1 — Implied Volatility ===")
market_price = 18.99   # Reliance option premium
iv = implied_vol_call_bsm(S0, K, T1, r_cc, q_cc, market_price)
print(f"Market premium (₹): {market_price:.2f}")
print(f"Implied Volatility: {iv*100:.2f}%")

# Plot price vs sigma
sigmas = np.linspace(0.05, 0.40, 200)
prices = [bsm_call_price(S0, K, T1, r_cc, s, q_cc) for s in sigmas]
plt.plot(sigmas, prices, label="BSM Call Price")
plt.axhline(market_price, color="red", linestyle="--", label="Market Price")
plt.axvline(iv, color="green", linestyle=":", label=f"IV = {iv*100:.2f}%")
plt.title("Part 1: Call Price vs Volatility")
plt.xlabel("Volatility (σ)")
plt.ylabel("Call Price (₹)")
plt.legend()
plt.show()

# ===================================================
# Part 2 — BSM vs Binomial pricing
# ===================================================
print("\n=== Part 2 — BSM vs Binomial Pricing ===")
sigma_hist = 0.174   # proxy historical volatility
bsm_price = bsm_call_price(S0, K, T1, r_cc, sigma_hist, q_cc)
undervaluation_bsm = (bsm_price - market_price)/market_price*100
crr_prices = [crr_binomial_call(S0, K, T1, r_cc, sigma_hist, q_cc, steps=n) for n in [150,200,300,340,400,500]]
steps = [150,200,300,340,400,500]

print(f"BSM price (σ={sigma_hist:.3f}): {bsm_price:.2f}  | Undervaluation: {undervaluation_bsm:.2f}%")
print("CRR prices:", dict(zip(steps, [round(p,2) for p in crr_prices])))

plt.plot(steps, crr_prices, marker="o", label="CRR price")
plt.axhline(bsm_price, linestyle="--", label="BSM price")
plt.axhline(market_price, linestyle=":", color="red", label="Market Price")
plt.title("Part 2: CRR Convergence to BSM")
plt.xlabel("Steps")
plt.ylabel("Price (₹)")
plt.legend()
plt.show()

# ===================================================
# Part 3 — Delta Hedging (Daily vs Weekly)
# ===================================================
print("\n=== Part 3 — Delta Hedging Simulation ===")
T2 = 60/365; n_steps=42; n_sims=2000; mu=0.10
S_paths = simulate_paths_gbm(S0, mu, iv, T2, n_steps, n_sims, seed=81)
res_daily  = buyers_hedge_pnl(S_paths, K, r_cc, q_cc, iv, T2, 'daily')
res_weekly = buyers_hedge_pnl(S_paths, K, r_cc, q_cc, iv, T2, 'weekly')
reduction = (res_weekly.std - res_daily.std)/res_weekly.std*100

print(f"Daily Hedge → mean={res_daily.mean:.2f}, std={res_daily.std:.2f}")
print(f"Weekly Hedge → mean={res_weekly.mean:.2f}, std={res_weekly.std:.2f}")
print(f"Std Dev Reduction: {reduction:.2f}%")

plt.hist(res_daily.pnl, bins=50, alpha=0.7)
plt.title("Part 3: P&L Distribution — Daily Hedge")
plt.show()

plt.hist(res_weekly.pnl, bins=50, alpha=0.7)
plt.title("Part 3: P&L Distribution — Weekly Hedge")
plt.show()

plt.bar(["Daily","Weekly"], [res_daily.std, res_weekly.std])
plt.title("Part 3: Std Dev Comparison")
plt.ylabel("Std Dev (₹)")
plt.show()
