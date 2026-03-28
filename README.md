
# Deep BSDE Solver for Bermudan Swaptions under the Hull-White Model

This project implements a deep learning-based backward BSDE solver for pricing Bermudan swaptions in the one-factor Hull-White short rate model, combining neural network approximation of the BSDE control process with an exact Hull-White Monte Carlo engine. The approach is inspired by [Wang et al. (2018)](https://ssrn.com/abstract=3214596) where a similar approach is carried out in the LIBOR-market model. Here we will use an interest rate model, the Hull-White model, to simulate the forward process of the FBSDE.

A Bermudan swaption grants the holder the right to enter into an interest rate swap at one of several exercise dates $T_E \in \lbrace T_E^1, \ldots, T_E^n \rbrace$. The underlying swap exchanges fixed payments at dates $T_1, \ldots, T_n$ with day count fractions $\tau_i = T_i - T_{i-1}$ against floating payments at dates 

$$\widetilde{T}_1, \ldots, \widetilde{T}_m, \qquad \widetilde{\tau}_j=  \widetilde{T}_j - \widetilde{T}_{j-1}.$$

The swaption payoff, forward swap rate, and swap annuity are

$$\begin{aligned}
H(T_E) &= \mathrm{An}(T_E)(S(T_E)-K)^+, \\
S(t) &= \frac{\sum_{j=1}^{m} \widetilde{\tau}_jL(t;\widetilde{T}_{j-1},\widetilde{T}_j)P(t,\widetilde{T}_j)}{\sum_{i=1}^{n} \tau_iP(t,T_i)}, \\
\mathrm{An}(t) &= \sum_{i=1}^{n} \tau_i P(t,T_i),
\end{aligned}$$

with zero-coupon bond prices $P(t,T)$, forward rates $L$, and strike $K$. Under the risk-neutral measure, the short rate $r_t$ follows Hull-White dynamics and the discounted swaption value $\widetilde{V}_t = V_t / B(t)$ satisfies a linear BSDE. Together they form the decoupled FBSDE system

$$\begin{aligned}
\mathrm{d}r_t &= (\theta(t) - a r_t)\mathrm{d}t + \sigma\mathrm{d}W_t, \\
\mathrm{d}\widetilde{V}_t &= \widetilde{Z}_t \mathrm{d}W_t, \\
\widetilde{V}_{T_E} &= \frac{H(T_E)}{B(T_E)}, \qquad r_0 \in \mathbb{R},
\end{aligned}$$

where $W$ is a Brownian motion, $a > 0$ is the mean reversion, $\sigma > 0$ the volatility, $\theta(t)$ is calibrated to the initial yield curve, $B(t) = \exp\left(\int_0^t r_s\mathrm{d}s\right)$ is the numeraire, and $\widetilde{Z}_t = \frac{Z_t}{B(t)}$ is the discounted control process. In integral notation, the system reads

$$\begin{aligned}
r_t &=r_0 e^{-at} +\int_0^t e^{-a(t-s)}\theta(s)\mathrm ds +\sigma \int_0^t e^{-a(t-s)}\mathrm dW_s, \\
\widetilde{V}_t &= \frac{H(T_E)}{B(T_E)} - \int_t^{T_E} \widetilde{Z}_s\mathrm{d}W_s.
\end{aligned}$$

The forward process is simulated exactly using the Gaussian transition density of the Hull-White model. On a time grid $\pi: 0 = t_0 < \cdots < t_N = T$ containing all Bermudan exercise dates, the exact one-step recursion is

$$\begin{aligned}
r_{t_{n+1}} = r_{t_n} e^{-a\Delta t_n} &+ f^M(0,t_{n+1}) - f^M(0,t_n)e^{-a\Delta t_n} \\
&+ \frac{\sigma^2}{2a^2}\left[(1-e^{-at_{n+1}})^2 - e^{-a\Delta t_n}(1-e^{-at_n})^2\right] \\
&+ \sqrt{\frac{\sigma^2}{2a}(1-e^{-2a\Delta t_n})}-\xi_n
\end{aligned}$$

with $\xi_n \sim \mathcal{N}(0,1)$ and $f^M(0,t)$ the market-implied instantaneous forward rate. The numeraire is approximated by the trapezoidal rule $s_{t_{n+1}} = s_{t_n} + \frac{1}{2}(r_{t_n}+r_{t_{n+1}})\Delta t_n$ with $B(t_n) = e^{s_{t_n}}$.

On the grid, the BSDE discretizes to the backward recursion

$$\widetilde{V}_{t_n}^\pi = \widetilde{V}_{t_{n+1}}^\pi - \widetilde{Z}_{t_n}^\pi\Delta W_{t_n}, \qquad \widetilde{V}_{t_N}^\pi = \frac{H(T_E)}{B(T_E)}.$$

The unknown control process $\widetilde{Z}_{t_n}^\pi$ is approximated by time-dependent neural networks $\varphi(\theta_n)$ at each grid point, so that

$$\widetilde{V}_{t_n}^{\pi,\theta} \simeq \widetilde{V}_{t_{n+1}}^{\pi,\theta} - \varphi_{\widetilde{Z}}(t_n, r_{t_n}\mid\theta_n)\Delta W_{t_n}.$$

For Bermudan swaptions, the backward recursion additionally incorporates the Snell envelope for optimal early exercise as

The simulation grid $\pi$ contains all exercise dates, i.e. $T_E^1, \ldots , T_E^n \in \pi$ for $i=1,\ldots , n$. The deep BSDE solver approximates the control process by neural networks $\varphi(\theta_n)$ for $n = 0, \dots, N-1$, so the backward recursion becomes

$$
\begin{aligned}
\widetilde{V}_{t_n}^{\pi,\theta} &= \begin{cases} \max\bigl(\widetilde{H}_{t_n}, \widetilde{V}_{t_{n+1}}^{\pi,\theta} - \widetilde Z_{t_n}^{\pi,\theta} \Delta W_{t_n}\bigr), & t_n \in \mathcal{T}_{\mathrm{Berm}}, \\
\widetilde{V}_{t_{n+1}}^{\pi,\theta} - \widetilde{Z}_{t_n}^{\pi,\theta} \Delta W_{t_n}, & \text{otherwise}, \end{cases} \\
&\simeq  \begin{cases} \max\bigl(\widetilde{H}_{t_n}, \widetilde{V}_{t_{n+1}}^{\pi,\theta} - \varphi_{\widetilde{Z}}(t_n, r_{t_n} \mid \theta_n) \Delta W_{t_n}\bigr), & t_n \in \mathcal{T}_{\mathrm{Berm}}, \\
\widetilde{V}_{t_{n+1}}^{\pi,\theta} - \varphi_{\widetilde{Z}}(t_n, r_{t_n} \mid \theta_n) \Delta W_{t_n}, & \text{otherwise}. \end{cases}
\end{aligned}
$$

Since the true discounted initial value $\widetilde{V}_0$ is deterministic, the deep BSDE solver determines the network parameters by minimizing the variance. Following Wang et al., the loss function is defined as

$$
L(\theta) = \bigl(\widetilde{V}_0^{\theta,\pi} - \mathbb{E}[\widetilde{V}_0^{\theta,\pi}]\bigr)^2,
$$

where $\widetilde{V}_0^{\theta,\pi}$ denotes the approximation of $\widetilde{V}_0$ induced by the neural networks on the grid $\pi$. The optimal network parameters $\theta^*$ are found by solving

$$
\theta^* \in \arg\min_{\theta \in \mathbb{R}^p} \mathbb{E}[L(\theta)] = \arg\min_{\theta \in \mathbb{R}^p} \mathrm{Var}(\widetilde{V}_0^{\theta,\pi})
$$

over $M$ simulated paths via stochastic gradient descent.







