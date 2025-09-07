# Option Pricing Model (Black–Scholes–Merton, CRR Binomial, Implied Volatility, Delta Hedging)

This repository implements a complete option pricing and hedging workflow using **Python**. It combines analytical and numerical methods for option valuation with a practical simulation of hedging strategies. The code is concise, reproducible, and structured to demonstrate both theoretical understanding and applied skills.

---

## Project Overview

This project addresses three key areas in derivatives modeling:

1. **Implied Volatility Estimation**

   * Extracts implied volatility from a market call option price using the **Black–Scholes–Merton (BSM) model with dividends**.
   * Implements a **bisection solver** to ensure convergence across a wide range of inputs.

2. **Option Pricing Models**

   * **BSM closed-form pricing** for European calls with continuous dividend yield.
   * **CRR Binomial Tree model** with dividend adjustment, demonstrating numerical convergence to BSM values after sufficient steps (≈ 340 in this case).

3. **Delta Hedging Simulation**

   * Simulates **Geometric Brownian Motion (GBM)** price paths.
   * Compares **weekly vs daily delta rebalancing** across 2,000 simulated paths.
   * Shows that **daily hedging significantly reduces the standard deviation** of hedged portfolio P\&L (≈ 54% reduction in this experiment).

---

## Why This Project Is Useful

* **For Learning**: Demonstrates the connection between closed-form pricing, discrete-time models, and risk management techniques.
* **For Finance Applications**: Provides a reproducible way to back out implied volatilities, cross-check models, and assess hedging strategies.
* **For Interviews/Recruitment**: Serves as strong proof of applied knowledge in options pricing, numerical methods, and portfolio risk reduction.
* **For Industry Practice**: Offers a lightweight implementation that can be extended to real market data and more complex derivatives.

---

## Repository Structure

* `requirements.txt` – Python dependencies.
* `run.py` – Single script covering:

  * BSM model pricing and delta.
  * Implied volatility solver.
  * CRR binomial tree implementation.
  * GBM simulation and delta-hedging comparison.
* `README.md` – Documentation and usage instructions.

---


## Sample Results

**Implied Volatility**

* Extracted \~17.32% from the given Reliance call option market price.

**Pricing Comparison**

* BSM vs CRR (n = 340): convergence within a few basis points.
* Undervaluation identified at \~0.54% using CRR vs BSM.

**Delta Hedging**

* Standard deviation of portfolio P\&L reduced by **53.93%** when switching from weekly to daily hedging.

---


