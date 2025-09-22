#!/usr/bin/env python3
"""
Option Pricing Model — Reliance Call (BSM, CRR, Implied Vol, Delta Hedging)

Author: (Your Name)
Description:
  - Reads a single-row CSV with S0, K, T_years, r_annual, q_annual, market_call_price
  - Computes implied vol via bisection, BSM price, CRR price (n_steps), and
    delta-hedged P&L distributions for weekly vs daily rebalancing.
  - Optionally saves plots and a JSON summary.

Usage:
  python run.py --input data/reliance_option_inputs.csv --plots \
    --n_steps 340 --n_paths 2000 --daily 252 --weekly 52 --seed 37
"""
import argparse, json, math, os
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

def norm_cdf(x): 
    return 0.5*(1.0 + math.erf(x/math.sqrt(2.0)))

def bsm_call_price(S,K,T,r,q,sigma):
    if T<=0 or sigma<=0:
        return max(0.0, S*math.exp(-q*T) - K*math.exp(-r*T))
    d1=(math.log(S/K)+(r-q+0.5*sigma*sigma)*T)/(sigma*math.sqrt(T))
    d2=d1 - sigma*math.sqrt(T)
    return S*math.exp(-q*T)*norm_cdf(d1) - K*math.exp(-r*T)*norm_cdf(d2)

def bsm_call_delta(S,K,T,r,q,sigma):
    if T<=0 or sigma<=0:
        return 1.0 if S>K else 0.0
    d1=(math.log(S/K)+(r-q+0.5*sigma*sigma)*T)/(sigma*math.sqrt(T))
    return math.exp(-q*T)*norm_cdf(d1)

def implied_vol_bisection(mkt,S,K,T,r,q,lo=1e-6,hi=5.0,tol=1e-8,iters=300):
    intrinsic=max(0.0, S*math.exp(-q*T) - K*math.exp(-r*T))
    if mkt<=intrinsic+1e-12:
        return 1e-6
    a,b=lo,hi
    for _ in range(iters):
        mid=0.5*(a+b)
        price=bsm_call_price(S,K,T,r,q,mid)
        if abs(price-mkt)<tol:
            return mid
        if price>mkt:
            b=mid
        else:
            a=mid
    return 0.5*(a+b)

def crr_call(S,K,T,r,q,sigma,steps):
    if steps<1:
        return bsm_call_price(S,K,T,r,q,sigma)
    dt=T/steps
    u=math.exp(sigma*math.sqrt(dt))
    d=1.0/u
    a=math.exp((r-q)*dt)
    p=(a-d)/(u-d)
    p=min(max(p,0.0),1.0)  # numerical guard
    prices=np.array([S*(u**j)*(d**(steps-j)) for j in range(steps+1)])
    payoffs=np.maximum(prices-K,0.0)
    disc=math.exp(-r*dt)
    for _ in range(steps,0,-1):
        payoffs=disc*(p*payoffs[1:]+(1-p)*payoffs[:-1])
    return payoffs[0]

def simulate_gbm_paths(S0,r,q,sigma,T,n_steps,n_paths,seed=None):
    if seed is not None:
        np.random.seed(seed)
    dt=T/n_steps
    Z=np.random.normal(size=(n_paths,n_steps))
    inc=((r-q)-0.5*sigma*sigma)*dt + sigma*math.sqrt(dt)*Z
    paths=np.zeros((n_paths,n_steps+1))
    paths[:,0]=S0
    paths[:,1:]=S0*np.exp(np.cumsum(inc,axis=1))
    return paths

def delta_hedge_pnl(paths,K,r,q,sigma,T,rebalance_steps):
    n_paths,n_cols=paths.shape
    n_steps_total=n_cols-1
    grid=np.linspace(0,n_steps_total,rebalance_steps+1,dtype=int)
    dt=T/n_steps_total
    pnl=np.zeros(n_paths)
    for i in range(n_paths):
        S_series=paths[i,:]
        V0=bsm_call_price(S_series[0],K,T,r,q,sigma)
        cash=V0
        stock_pos=0.0
        for k in range(len(grid)-1):
            idx=grid[k]
            t_rem=max(T-idx*dt,1e-12)
            d_now=bsm_call_delta(S_series[idx],K,t_rem,r,q,sigma)
            d_change=(-d_now)-stock_pos
            cash-=d_change*S_series[idx]
            stock_pos+=d_change
            next_idx=grid[k+1]
            dt_seg=(next_idx-idx)*dt
            cash*=math.exp(r*dt_seg)
        ST=S_series[-1]
        payoff=max(ST-K,0.0)
        pnl[i]=cash + stock_pos*ST - payoff
    return pnl

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",type=str,default="data/reliance_option_inputs.csv")
    ap.add_argument("--n_steps",type=int,default=340)
    ap.add_argument("--n_paths",type=int,default=2000)
    ap.add_argument("--daily",type=int,default=252)
    ap.add_argument("--weekly",type=int,default=52)
    ap.add_argument("--plots",action="store_true")
    ap.add_argument("--seed",type=int,default=None)
    ap.add_argument("--out_json",type=str,default=None)
    args=ap.parse_args()

    # Read input
    df=pd.read_csv(args.input)
    if len(df)!=1:
        raise ValueError("Input CSV should contain exactly one row for this run.")
    row=df.iloc[0]
    S0=float(row["S0"]); K=float(row["K"]); T=float(row["T_years"])
    r=float(row["r_annual"]); q=float(row["q_annual"]); mkt=float(row["market_call_price"])

    # Pricing & IV
    iv=implied_vol_bisection(mkt,S0,K,T,r,q)
    bsm=bsm_call_price(S0,K,T,r,q,iv)
    crr=crr_call(S0,K,T,r,q,iv,args.n_steps)
    diff_bps=(crr-bsm)/bsm*1e4

    # Hedging
    paths=simulate_gbm_paths(S0,r,q,iv,T,n_steps=252,n_paths=args.n_paths,seed=args.seed)
    pnl_daily=delta_hedge_pnl(paths,K,r,q,iv,T,rebalance_steps=args.daily)
    pnl_weekly=delta_hedge_pnl(paths,K,r,q,iv,T,rebalance_steps=args.weekly)
    sd_daily=float(np.std(pnl_daily)); sd_weekly=float(np.std(pnl_weekly))
    reduction=(1 - sd_daily/sd_weekly)*100.0 if sd_weekly>0 else float("nan")

    out={"inputs":{"S0":S0,"K":K,"T":T,"r":r,"q":q,"market":mkt},
         "implied_vol":iv,"bsm_price":bsm,"crr_price":crr,"crr_diff_bps":diff_bps,
         "hedging":{"std_daily":sd_daily,"std_weekly":sd_weekly,"reduction_pct":reduction}}
    print(json.dumps(out, indent=2))

    if args.plots:
        os.makedirs("figures",exist_ok=True)
        plt.figure(); plt.hist(pnl_daily,bins=50); plt.title("Delta-Hedged P&L Distribution (Daily Rebalance)"); plt.xlabel("P&L (INR)"); plt.ylabel("Frequency"); plt.savefig("figures/hist_pnl_daily.png",bbox_inches="tight"); plt.close()
        plt.figure(); plt.hist(pnl_weekly,bins=50); plt.title("Delta-Hedged P&L Distribution (Weekly Rebalance)"); plt.xlabel("P&L (INR)"); plt.ylabel("Frequency"); plt.savefig("figures/hist_pnl_weekly.png",bbox_inches="tight"); plt.close()
        grid=[10,20,40,80,160,240,300,args.n_steps,400,600]; vals=[crr_call(S0,K,T,r,q,iv,g) for g in grid]
        plt.figure(); plt.plot(grid,vals,marker="o"); plt.axhline(bsm,linestyle="--"); plt.title("CRR Convergence to BSM (Call Price)"); plt.xlabel("Steps (n)"); plt.ylabel("Price (INR)"); plt.savefig("figures/crr_convergence.png",bbox_inches="tight"); plt.close()
        sigmas=np.linspace(0.05,0.40,200); prices=[bsm_call_price(S0,K,T,r,q,s) for s in sigmas]
        plt.figure(); plt.plot(sigmas,prices); plt.axhline(mkt,linestyle="--"); plt.title("BSM Price vs Volatility with Market Price"); plt.xlabel("Sigma"); plt.ylabel("Price (INR)"); plt.savefig("figures/price_vs_vol.png",bbox_inches="tight"); plt.close()

    if args.out_json:
        with open(args.out_json,"w") as f: json.dump(out,f,indent=2)

if __name__=="__main__":
    main()
