# Deep BSDE Solver for Bermudan Swaptions under the Hull-White Model

A deep learning-based backward stochastic differential equation (BSDE) solver for pricing Bermudan swaptions in the one-factor Hull-White short rate model.

## Overview

This project implements the **deep BSDE backward solver** for pricing Bermudan swaptions, combining neural network approximation of the BSDE control process with an exact Hull-White Monte Carlo simulation engine. The approach is inspired by [Wang et al. (2018)](https://ssrn.com/abstract=3214596) and formulated specifically for the Hull-White interest rate framework.

## Mathematical Framework

### The Forward-Backward SDE System

Under the risk-neutral measure **Q**, the short rate follows the Hull-White dynamics and the discounted swaption value satisfies a linear BSDE. Together they form a decoupled forward-backward SDE (FBSDE):

**Forward SDE** (short rate):

$$dr_t = (\theta(t) - a \, r_t) \, dt + \sigma \, dW_t^{\mathbb{Q}}, \qquad r_0 \in \mathbb{R}$$

**Backward SDE** (discounted value):

$$d\widetilde{V}_t = \widetilde{Z}_t \, dW_t^{\mathbb{Q}}, \qquad \widetilde{V}_{T_E} = \frac{H(T_E)}{B(T_E)}$$

where:
- $r_t$ is the short rate, $a > 0$ the mean reversion, $\sigma > 0$ the volatility, and $\theta(t)$ is calibrated to the initial yield curve
- $\widetilde{V}_t = V_t / B(t)$ is the discounted option value with $B(t) = \exp(\int_0^t r_s \, ds)$
- $\widetilde{Z}_t = Z_t / B(t)$ is the discounted control process
- $H(T_E) = \text{An}(T_E) \cdot (S(T_E) - K)^+$ is the swaption payoff

In integral form, the system reads:

$$r_t = r_0 + \int_0^t (\theta(s) - a \, r_s) \, ds + \int_0^t \sigma \, dW_s^{\mathbb{Q}}$$

$$\widetilde{V}_t = \frac{H(T_E)}{B(T_E)} - \int_t^{T_E} \widetilde{Z}_s \, dW_s^{\mathbb{Q}}$$

### Hull-White Monte Carlo Simulation

The forward process is simulated exactly (not via Euler-Maruyama) using the Gaussian transition density of the Hull-White model. On a time grid $\pi: 0 = t_0 < t_1 < \cdots < t_N = T$, the exact one-step recursion is:

$$r_{t_{n+1}} = r_{t_n} e^{-a \Delta t_n} + f^M(0, t_{n+1}) - f^M(0, t_n) e^{-a \Delta t_n} + \frac{\sigma^2}{2a^2} \left[(1 - e^{-a t_{n+1}})^2 - e^{-a \Delta t_n}(1 - e^{-a t_n})^2 \right] + \sqrt{\frac{\sigma^2}{2a}(1 - e^{-2a \Delta t_n})} \, \xi_n$$

where $\xi_n \sim \mathcal{N}(0,1)$ and $f^M(0,t)$ is the market-implied instantaneous forward rate. The numeraire is approximated via the trapezoidal rule: $s_{t_{n+1}} = s_{t_n} + \frac{1}{2}(r_{t_n} + r_{t_{n+1}}) \Delta t_n$ with $B(t_n) = e^{s_{t_n}}$.

### Deep BSDE Backward Solver

The key idea is to approximate the unknown control process $\widetilde{Z}_{t_n}$ at each grid point by a **time-dependent neural network**:

$$\varphi_{\widetilde{Z}}(t_n, r_{t_n} \mid \theta_n) \approx \widetilde{Z}_{t_n}$$

The backward recursion for the Bermudan swaption value on the grid incorporates the **Snell envelope** for optimal early exercise:

- At exercise dates $t_n \in \mathcal{T}_{\text{Berm}}$: $\widetilde{V}_{t_n} = \max(\widetilde{H}_{t_n}, \, \widetilde{V}_{t_{n+1}} - \varphi_{\widetilde{Z}}(t_n, r_{t_n} \mid \theta_n) \Delta W_{t_n})$
- Otherwise: $\widetilde{V}_{t_n} = \widetilde{V}_{t_{n+1}} - \varphi_{\widetilde{Z}}(t_n, r_{t_n} \mid \theta_n) \Delta W_{t_n}$

Since the true initial value $\widetilde{V}_0$ is deterministic, the network parameters are optimized by **variance minimization**:

$$\theta^* \in \arg\min_{\theta} \text{Var}(\widetilde{V}_0^{\theta, \pi})$$

The empirical loss is computed over $M$ Monte Carlo paths and minimized via stochastic gradient descent.

### Algorithm Summary

1. **Forward pass**: Simulate $M$ short rate paths using the exact Hull-White recursion
2. **Backward pass**: Starting from the terminal payoff, propagate backwards using the neural network approximation and apply the exercise condition at Bermudan dates
3. **Loss**: Compute the sample variance of the initial value estimates $\widetilde{V}_0^{(m)}$
4. **Update**: Gradient descent on the network parameters
5. **Repeat** for $K$ epochs

## Numerical Experiments

The solver is tested on Bermudan receiver swaptions with:
- **19 annual exercise dates** on a 20-year swap
- Hull-White model calibrated to a market yield curve
- Comparisons against **PDE solver** and **American Monte Carlo (AMC)** benchmarks

Key findings:
- The BSDE solver achieves pricing accuracy within 1-5% of PDE benchmarks
- A `softplus` or `ReLU` activation with architecture `(16, 32, 64, 32, 16)` yields the best results
- Grid refinement has a stronger impact on the BSDE solver than on AMC due to the time-dependent network structure
- Approximately 20,000-50,000 Monte Carlo paths are required for stable convergence

## Requirements

- Python 3.8+
- PyTorch
- NumPy, SciPy
- Julia DiffFusion library (for Hull-White model calibration)

## References

- Wang et al. (2018). *Deep Learning-Based BSDE Solver for Libor Market Model with Applications to Bermudan Swaption Pricing.* SSRN.
- Gao et al. (2023). *Convergence of the Backward Deep BSDE Method.* arXiv:2210.04118.
- Brigo & Mercurio (2006). *Interest Rate Models: Theory and Practice.* Springer.
- Hull & White (1990). *Pricing Interest-Rate-Derivative Securities.* Review of Financial Studies.
