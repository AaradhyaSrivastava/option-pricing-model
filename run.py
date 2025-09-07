from __future__ import annotations
if T_rem <= 0:
break
delta = bs_call_delta(S[t-1], K, r, q, sigma, T_rem)
# trade shares to new delta
dS = delta - delta_prev
cash -= dS * S[t-1]
# accrue risk‑free on cash over this period
cash *= math.exp(r * ( (t - t_prev)*dt ))
delta_prev = delta
t_prev = t
# expiry
payoff = max(S[-1]-K, 0.0)
portfolio = cash + delta_prev*S[-1] - payoff
pnl[i] = portfolio
return float(np.std(pnl))


# ------------------------ CLI and Orchestration ------------------------


def main():
ap = argparse.ArgumentParser()
ap.add_argument('--ticker', default='RIL')
ap.add_argument('--S', type=float, default=2950.0)
ap.add_argument('--K', type=float, default=3000.0)
ap.add_argument('--r', type=float, default=0.072) # annual risk‑free (cont comp approx)
ap.add_argument('--q', type=float, default=0.0034) # dividend yield
ap.add_argument('--T', type=float, default=0.25) # years to expiry
ap.add_argument('--market-price', type=float, default=120.0)
ap.add_argument('--n-steps', type=int, default=340)
ap.add_argument('--paths', type=int, default=2000)
ap.add_argument('--seed', type=int, default=42)
args = ap.parse_args()


# 1) Implied volatility from market price
iv = implied_vol_call_bisect(args.market_price, args.S, args.K, args.r, args.q, args.T)
print(f"Implied volatility: {iv*100:.2f}%")


# 2) Cross‑check BSM vs CRR binomial and report convergence at n_steps
bsm_p = bs_call_price(args.S, args.K, args.r, args.q, iv, args.T)
crr_p = crr_call_price(args.S, args.K, args.r, args.q, iv, args.T, args.n_steps)
diff = (crr_p - bsm_p)
print(f"BSM price: {bsm_p:.4f} | CRR(n={args.n_steps}) price: {crr_p:.4f} | diff: {diff:.4f}")


# 3) Delta‑hedging risk comparison: weekly vs daily
# choose calendar: 252 trading days/year
steps_daily = int(252*args.T)
steps_weekly = max(1, steps_daily//5)


params = GBMParams(S0=args.S, r=args.r, q=args.q, sigma=iv, T=args.T)
S_paths = simulate_gbm_paths(params, steps_daily, args.paths, seed=args.seed)


std_weekly = delta_hedge_stddev(S_paths, args.K, args.r, args.q, iv, args.T, rebalance_steps=steps_weekly)
std_daily = delta_hedge_stddev(S_paths, args.K, args.r, args.q, iv, args.T, rebalance_steps=steps_daily)


if std_weekly > 0:
reduction = 100.0*(std_weekly - std_daily)/std_weekly
else:
reduction = 0.0
print(f"StdDev (weekly): {std_weekly:.4f} | StdDev (daily): {std_daily:.4f} | reduction: {reduction:.2f}%")


if __name__ == '__main__':
main()
