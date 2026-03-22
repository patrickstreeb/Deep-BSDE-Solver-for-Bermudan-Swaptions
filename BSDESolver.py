import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd

from scipy.optimize import brentq
from tqdm import tqdm
import QuantLib as ql
import contextlib
import io
from tqdm.auto import trange

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("plots", exist_ok=True)

# --- project modules from  ---
from src.yieldcurve import YieldCurve
from src.hull_white_model import HullWhiteModel
from src.swaption import create_swaption
from src.bermudan_option import bermudan_option_npv
from src.monte_carlo_simulation import MonteCarloSimulation

from src.methods.payoffs import CouponBond
from src.methods.amc_solver import AmcSolver, StateVariableControls
from src.methods.pde_solver import PdeSolver



# ===============================================================================
# Deep Backward BSDE Solver for Bermudan Swaption Pricing in the Hull-White model
# ===============================================================================
from src.monte_carlo_simulation import MonteCarloSimulation

class BsdeBackwardSolver(nn.Module):
    """
    Deep Backward BSDE solver for Bermudan swaption pricing in the Hull–White model.

    This class implements a time-discretized, discounted backward stochastic
    differential equation with reflection at exercise dates. The solution is
    represented in numeraire units and learned by minimizing the empirical
    variance of the initial value.

    Main methods
    ------------
    __init__()
        Initializes the solver, builds one neural network per time step for the
        control process, and sets up the optimizer.

    train_epoch()
        Performs a single training epoch by simulating Monte Carlo paths,
        propagating the BSDE backward in time, applying the early-exercise
        constraint, and updating the neural networks.

    fit()
        Repeatedly calls train_epoch to train the solver for a given number of
        epochs and paths, while storing loss and price trajectories.

    evaluate()
        Computes an out-of-sample price estimate and the distribution of Y_0
        using a fresh Monte Carlo simulation without gradient tracking.

    Helper methods
    --------------
    _make_activation()
        Returns the activation function specified by name.

    _intrinsic_value()
        Computes the intrinsic value of the remaining swap at a given exercise
        date.
    """

    def __init__(self, 
                 hw_model, 
                 times, 
                 payoffs, 
                 exercise_indices,
                 payer=False,
                 architecture=(16, 32, 64, 32, 16),
                 activation="softplus",
                 lr=0.1, 
                 seed=1234):

        super().__init__()
        self.hw      = hw_model
        self.times   = np.asarray(times, float)
        self.payoffs = payoffs
        self.exercise_indices = list(exercise_indices)
        self.payer   = bool(payer)

        torch.manual_seed(seed); np.random.seed(seed)

        self.dim_x = hw_model.size()
        self.dim_d = hw_model.factors()

        self.architecture = tuple(architecture)
        self.activation   = str(activation).lower()

        # build one NN per time step
        act = self._make_activation(self.activation)
        self.models = nn.ModuleList()
        for _ in range(len(self.times) - 1):
            layers, in_dim = [], self.dim_x
            for h in self.architecture:
                layers += [nn.Linear(in_dim, h), act]
                in_dim = h
            layers += [nn.Linear(in_dim, self.dim_d)]
            self.models.append(nn.Sequential(*layers).to(device))

        # optimizer over all nets
        self.opt = torch.optim.Adam(self.parameters(), lr=lr)
        self.losses, self.y0_vals, self.price_var = [], [], []

    # ---------- helpers ----------
    def _make_activation(self, name):
        name = name.lower()
        if name == "relu":     return nn.ReLU()
        if name == "tanh":     return nn.Tanh()
        if name == "sigmoid":  return nn.Sigmoid()
        if name == "softplus": return nn.Softplus()
        raise ValueError(f"Unknown activation '{name}'")

    def _intrinsic_value(self, payoff_idx, X_hw):
        """Intrinsic value at exercise time of payoffs[payoff_idx]."""
        B = np.asarray(self.payoffs[payoff_idx].at(X_hw), dtype=np.float64).reshape(-1)
        return B

    # ---------- one training epoch ----------
    def train_epoch(self, n_paths, epoch_seed=None):
        sim = MonteCarloSimulation(self.hw, self.times, n_paths,
                                   seed=(int(epoch_seed) if epoch_seed is not None else None),
                                   showProgress=False)

        X  = torch.from_numpy(np.transpose(sim.X,  (2, 0, 1))).float().to(device)
        dW = torch.from_numpy(np.transpose(sim.dW, (2, 0, 1))).float().to(device)
        batch, N, _ = X.shape # batch = Batchsize, N = Gridsize

        u = torch.zeros(batch, device=device) + 0.00001


        for i in reversed(range(N-1)):
            t_i = float(self.times[i])
            X_i = X[:, i, :]                       # (B, dim_x)

            Z = self.models[i](X_i).view(batch, self.dim_d)
            u = u - torch.sum(Z * dW[:, i, :], dim=1)

            if i in self.exercise_indices:
                j = self.exercise_indices.index(i)
                X_i_T = X[:, i, :].detach().cpu().numpy().T.copy()
                H = self._intrinsic_value(j, X_i_T) # terminal condition
                N_i = np.asarray(self.hw.numeraire(X_i_T, t_i), dtype=np.float64) #numeraire
                H_discounted = torch.from_numpy(H / np.clip(N_i, 1e-12, None)).float().to(device)
                u = torch.maximum(u, H_discounted)


        mu0 = torch.mean(u)
        var0 = (1 / batch) * torch.mean((u - mu0) ** 2)
        loss = var0 

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        # logging
        X0_hw = X[:, 0, :].detach().cpu().numpy().T.copy()
        N0 = np.asarray(self.hw.numeraire(X0_hw, 0.0), dtype=np.float64)
        y0_paths = u.detach().cpu().numpy() * N0
        price_est = float(np.mean(y0_paths))
        self.y0_vals.append(price_est)
        self.losses.append(float(loss.detach().cpu().item()))
        self.price_var.append(float(np.var(y0_paths)))

        return loss, price_est

    def fit(self, n_epochs, n_paths):
        from tqdm.auto import trange

        bar = trange(int(n_epochs), desc="BSDE training", leave=True)
        for epoch in bar:
            loss, price_estimate = self.train_epoch(n_paths)

            bar.set_postfix(
                loss=f"{loss.item():.2e}",
                Y0=f"{price_estimate:.4f}"
            )


    def evaluate(self, n_paths, seed=None):
        sim = MonteCarloSimulation(self.hw, self.times, n_paths, seed=seed, showProgress=False)

        X  = torch.from_numpy(np.transpose(sim.X,  (2, 0, 1))).float().to(device)
        dW = torch.from_numpy(np.transpose(sim.dW, (2, 0, 1))).float().to(device)
        batch, N, _ = X.shape

        with torch.no_grad():
            u = torch.zeros(batch, device=device)
            for i in reversed(range(N-1)):
                t_i = float(self.times[i])
                X_i = X[:, i, :]
                Z = self.models[i](X_i).view(batch, self.dim_d)
                u = u - torch.sum(Z * dW[:, i, :], dim=1)

                if i in self.exercise_indices:
                    j = self.exercise_indices.index(i)
                    X_i_T = X[:, i, :].cpu().numpy().T.copy()
                    H = self._intrinsic_value(j, X_i_T)
                    N_i = np.asarray(self.hw.numeraire(X_i_T, t_i), dtype=np.float64)
                    H_discounted = torch.from_numpy(H / np.clip(N_i, 1e-12, None)).float().to(device)
                    u = torch.maximum(u, H_discounted)

            X0_hw = X[:, 0, :].cpu().numpy().T.copy()
            N0 = np.asarray(self.hw.numeraire(X0_hw, 0.0), dtype=np.float64)
            y0_paths = u.cpu().numpy() * N0
            price_est = float(np.mean(y0_paths))

        return price_est, y0_paths




# ---------- Calibration to co-terminal European swaptions (normal vols) ----------------
class Calibration:

    @staticmethod
    def model_from_swaptions(euro_swaptions, yield_curve, mean_reversion):
        details  = [s.bond_option_details() for s in euro_swaptions]
        ref_npv  = [s.npv_via_bachelier() for s in euro_swaptions]
        ref_vega = [s.vega() for s in euro_swaptions]

        vol_times = np.array([d['expiry_time'] for d in details])
        vol_vals  = np.zeros_like(vol_times)

        for k in range(len(euro_swaptions)):
            def obj(sig):
                vol_vals[k:] = sig
                mdl = HullWhiteModel(yield_curve, mean_reversion, vol_times, vol_vals)
                mnpv = mdl.coupon_bond_option(
                    details[k]['expiry_time'],
                    details[k]['pay_times'],
                    details[k]['cash_flows'],
                    details[k]['strike_price'],
                    details[k]['call_or_put']
                )
                return (mnpv - ref_npv[k]) / ref_vega[k]

            vg = euro_swaptions[k].normalVolatility
            vol_vals[k] = brentq(obj, 0.1 * vg, 5.0 * vg, xtol=1e-6)

        return HullWhiteModel(yield_curve, mean_reversion, vol_times, vol_vals)

    @staticmethod
    def nearest_index(grid, t):
        i = np.searchsorted(grid, t)
        return i - 1 if (i > 0 and (i == len(grid) or abs(t - grid[i-1]) < abs(t - grid[i]))) else i








# ---------- helper functions for plotting ----------------
# 1) plot_yield_and_discount_curve -> 1_yieldcurve.png
# 2) plot_y0_distributions
# 3) train_bsde_with_snapshots
# 4) plot_y0_snapshots_2x2
# 5) train_bsde_different_paths
# 6) plot_bsde_training_diagnostics
# 7) bond_option_details
# 8) bermudan_pricing_analysis
# 9) plot_bsde_training(solvers, filename): (for network depth, activation and node analysis)



class PlotHelpers:

    def plot_yield_and_discount_curve(
        yieldCurve,
        T_min=0.1,
        T_max=30.0,
        n_points=100,
        figsize=(12, 4),
        filename="plotsNEU/1_yieldcurve.png",
    ):
        """
        Plot zero-coupon yield curve and discount factors.
        """
        T_grid = np.linspace(T_min, T_max, n_points)

        discounts = np.array([yieldCurve.discount(T) for T in T_grid])
        zero_yields = -np.log(discounts) / T_grid

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # ---- Zero yields
        axes[0].plot(T_grid, zero_yields * 100, lw=1.8)
        axes[0].set_title("Yield Curve")
        axes[0].set_xlabel(r"Maturity $T$ (years)")
        axes[0].set_ylabel(r"Yield (\%)")
        axes[0].grid(True)

        # ---- Discount factors
        axes[1].plot(T_grid, discounts, lw=1.8)
        axes[1].set_title(r"Discount Factors $P(0,T)$")
        axes[1].set_xlabel(r"Maturity $T$ (years)")
        axes[1].set_ylabel("Discount Factor")
        axes[1].grid(True)

        plt.tight_layout()
        plt.savefig(filename, dpi=300)

        return fig, axes


    def plot_y0_distributions(
        y0_left,
        y0_right,
        mean_left,
        mean_right,
        label_left="10 epochs",
        label_right="1500 epochs",
        bins=30,
        density=True,
        alpha=0.7,
        color="steelblue",
        edgecolor="k",
        figsize=(12, 4),
        filename=None,
        dpi=300,
        sharey=True,
    ):
        """
        Plot side-by-side histograms of BSDE Y0 pathwise distributions.
        """

        y0_left = np.asarray(y0_left)
        y0_right = np.asarray(y0_right)

        xmin = min(y0_left.min(), y0_right.min())
        xmax = max(y0_left.max(), y0_right.max())

        fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=sharey)

        # left panel
        axes[0].hist(
            y0_left, bins=bins, density=density,
            alpha=alpha, color=color, edgecolor=edgecolor
        )
        axes[0].axvline(mean_left, color="red", linestyle="--",
                        label=f"mean={mean_left:.4f}")
        axes[0].set_title(f"BSDE $Y_0$ distribution ({label_left})")
        axes[0].set_xlabel("pathwise $Y_0$")
        axes[0].set_ylabel("density")
        axes[0].set_xlim(xmin, xmax)
        axes[0].legend()

        # right panel
        axes[1].hist(
            y0_right, bins=bins, density=density,
            alpha=alpha, color=color, edgecolor=edgecolor
        )
        axes[1].axvline(mean_right, color="red", linestyle="--",
                        label=f"mean={mean_right:.4f}")
        axes[1].set_title(f"BSDE $Y_0$ distribution ({label_right})")
        axes[1].set_xlabel("pathwise $Y_0$")
        axes[1].set_xlim(xmin, xmax)
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(filename, dpi=dpi)

        return fig, axes



    def train_bsde_with_snapshots(
        bsde,
        n_paths_train,
        n_paths_eval,
        snapshot_epochs,
        seed_base=42,
    ):
        """
        Train BSDE solver and collect Y0 distributions
        at specified epochs.
        """

        snapshot_epochs = sorted(snapshot_epochs)
        results = {}
        max_epoch = snapshot_epochs[-1]

        bar = trange(
            1,
            max_epoch + 1,
            desc="BSDE training (snapshots)",
            leave=True
        )

        for epoch in bar:
            loss, price_estimate = bsde.train_epoch(n_paths=n_paths_train)

            # live info (same style as fit)
            bar.set_postfix(
                loss=f"{loss.item():.2e}",
                Y0=f"{price_estimate:.4f}"
            )

            if epoch in snapshot_epochs:
                price, y0_paths = bsde.evaluate(
                    n_paths=n_paths_eval,
                    seed=seed_base + epoch
                )

                results[epoch] = {
                    "mean": price,
                    "std": float(np.std(y0_paths)),
                    "y0": y0_paths.copy(),
                }

        return results



    def plot_y0_snapshots_2x2(
        results,
        bins=30,
        xlim=(-6000, 6000),
        ylim=(0, 1),
        title_prefix="Snapshot",
        figsize=(12, 10),
        filename=None,
    ):
        """
        Plot a 2x2 grid of Y0 distributions.
        Keys of `results` are used as labels (epochs, paths, etc.).
        """

        labels = list(results.keys())
        fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=True, sharey=True)
        axes = axes.flatten()

        for ax, label in zip(axes, labels):
            y0 = results[label]["y0"]
            mean = results[label]["mean"]

            ax.hist(y0, bins=bins, density=True, alpha=0.75, edgecolor="k",)
            ax.axvline(mean, color="red", linestyle="--", linewidth=1.5)
            ax.set_title(f"{title_prefix}: {label}")
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.grid(True, alpha=0.3)

        fig.suptitle(r"BSDE $Y_0$ Distributions", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(filename, dpi=300)

        return fig, axes


    @staticmethod
    def train_bsde_different_paths(
        hw,
        t_grid,
        underlying_payoffs,
        ex_idx,
        n_epochs,
        path_grid,
        architecture,
        activation,
        lr,
        seed,
    ):
        """
        Train BSDE solver for different numbers of Monte Carlo paths.

        Returns
        -------
        results : dict
            keys   : number of paths
            values : dict with fields 'paths', 'mean', 'std', 'y0_paths'
        """
        results = {}

        for n_paths in path_grid:
            # print(f"[Path sweep] Training with {n_paths} paths")

            bsde = BsdeBackwardSolver(
                hw_model=hw,
                times=t_grid,
                payoffs=underlying_payoffs,
                exercise_indices=ex_idx,
                payer=False,
                architecture=architecture,
                activation=activation,
                lr=lr,
                seed=seed,
            )

            bsde.fit(n_epochs=n_epochs, n_paths=n_paths)

            price, y0_paths = bsde.evaluate(n_paths=n_paths, seed=seed)

            results[n_paths] = {
                "paths": n_paths,
                "mean": float(np.mean(y0_paths)),
                "std":  float(np.std(y0_paths)),
                "y0": y0_paths,
            }

        return results


    def plot_bsde_training_diagnostics(
        bsde,
        npv_amc,
        amc_degree=None,
        npv_pde=None,
        figsize=(10, 4),
        filename=None,
    ):

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # -------- Left panel: training objective
        if hasattr(bsde, "losses") and len(bsde.losses) > 0:
            axes[0].plot(bsde.losses, label=r"Training loss $\mathrm{Var}(u_0)$")
        axes[0].set_title("BSDE Training Objective")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Variance")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        # -------- Right panel: price convergence
        if hasattr(bsde, "y0_vals") and len(bsde.y0_vals) > 0:
            axes[1].plot(bsde.y0_vals, label=r"BSDE $Y_0$")

        label_amc = "AMC"
        if amc_degree is not None:
            label_amc += f" (deg={amc_degree})"

        axes[1].axhline(
            npv_amc,
            color="red",
            linestyle=":",
            linewidth=1.8,
            label=label_amc,
        )

        if npv_pde is not None:
            axes[1].axhline(
                npv_pde,
                color="blue",
                linestyle="--",
                linewidth=1.5,
                label="PDE",
            )

        axes[1].set_title("Price Approximation")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Price")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

        plt.tight_layout()

        if filename is not None:
            plt.savefig(filename, dpi=300)

        return fig, axes




    def run_grid_size_experiment(
        hw,
        exercise_times,
        underlying_payoffs,
        T_max,
        grid_sizes,
        n_epochs,
        n_paths,
        architecture,
        activation,
        lr,
        seed,
    ):
        """
        Run AMC and BSDE pricing for different time-grid sizes.

        Returns
        -------
        results : list of dict
            Each dict contains grid_size, price_bsde, price_amc
        """
        results = []

        for n_base in grid_sizes:
            base_grid = np.linspace(0.0, T_max, n_base)
            t_grid = np.unique(np.sort(np.concatenate([base_grid, exercise_times])))

            ex_idx = [
                np.where(np.isclose(t_grid, t, rtol=0, atol=1e-12))[0][0]
                for t in exercise_times
            ]

            # ---------- AMC ----------
            amc_method = AmcSolver(
                simulation=MonteCarloSimulation(
                    hw, t_grid, n_paths=n_paths, seed=seed
                ),
                max_polynomial_degree=2,
                split_ratio=0.25,
                controls=StateVariableControls(),
            )

            price_amc = bermudan_option_npv(
                exercise_times,
                underlying_payoffs,
                amc_method,
                showProgress=False,
            )

            # ---------- BSDE ----------
            bsde = BsdeBackwardSolver(
                hw_model=hw,
                times=t_grid,
                payoffs=underlying_payoffs,
                exercise_indices=ex_idx,
                payer=False,
                architecture=architecture,
                activation=activation,
                lr=lr,
                seed=seed,
            )

            bsde.fit(n_epochs=n_epochs, n_paths=n_paths)
            price_bsde = float(bsde.y0_vals[-1])

            results.append({
                "grid_size": n_base,
                "price_bsde": price_bsde,
                "price_amc": price_amc,
            })

        return results


    def show_grid_results(grid_results, filename):
        """
        display grid-size result table and plot.
        """

        df = pd.DataFrame(grid_results)
        display(df.round(1))

        x = df["grid_size"].to_numpy(dtype=float)
        x_dense = np.linspace(x.min(), x.max(), 400)
        y_bsde = np.interp(x_dense, x, df["price_bsde"])
        y_amc  = np.interp(x_dense, x, df["price_amc"])

        plt.figure(figsize=(8, 4.5))
        plt.plot(x_dense, y_bsde, label=f"BSDE", linewidth=1.8)
        plt.plot(x_dense, y_amc, label=f"AMC", linestyle="--", linewidth=1.5)
        plt.scatter(x, df["price_bsde"], s=18)
        plt.scatter(x, df["price_amc"],  s=18)
        plt.xlabel("base grid size (# of points)")
        plt.ylabel("price")
        plt.ylim(400,800)
        plt.legend()
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.show()

        return df


    def bond_option_details(european_swaptions, model):
        details  = [ s.bond_option_details() for s in european_swaptions ]
        exercise_times = np.array([ d['expiry_time'] for d in details ])
        underlying_payoffs = [
            CouponBond(model, d['expiry_time'], d['pay_times'], d['cash_flows'] * d['call_or_put'])
            for d in details
        ]
        return {
            'exercise_times' : exercise_times,
            'underlying_payoffs' : underlying_payoffs,
        }


    def bermudan_pricing_analysis(
        maturity,
        strike,
        rate,
        market_volatility,
        mean_reversion,
        show_plots=True,
        plot_save_path=None,
        # --- BSDE params ---
        bsde_epochs=400,
        bsde_paths=50000,
        bsde_hidden=(32, 64, 32),
        bsde_lr=0.1,
        seed=1234,
    ):

        # ---------------- Yield curve & Europeans ----------------
        yield_curve = YieldCurve(['70y'], [rate])

        expiry_terms = [f"{e}y" for e in range(1, maturity)]
        swap_terms   = [f"{maturity - e}y" for e in range(1, maturity)]

        swaptions = [
            create_swaption(
                e, s, yield_curve, yield_curve,
                strike=strike,
                normalVolatility=market_volatility,
                payerOrReceiver=ql.VanillaSwap.Receiver
            )
            for e, s in zip(expiry_terms, swap_terms)
        ]

        model = Calibration.model_from_swaptions(swaptions, yield_curve, mean_reversion)

        option = PlotHelpers.bond_option_details(swaptions, model)
        exercise_times = option["exercise_times"]
        underlying_payoffs = option["underlying_payoffs"]

        # ---------------- AMC reference ----------------
        T_max = max(d['pay_times'][-1] for d in [s.bond_option_details() for s in swaptions])
        base_grid = np.linspace(0.0, T_max, 102)
        t_grid = np.unique(np.sort(np.concatenate([base_grid, exercise_times])))

        amc_method = AmcSolver(
            simulation=MonteCarloSimulation(model, t_grid, n_paths=bsde_paths, seed=seed),
            max_polynomial_degree=2,
            split_ratio=0.25,
            controls=StateVariableControls()
        )

        berm_npv_amc = bermudan_option_npv(
            exercise_times,
            underlying_payoffs,
            amc_method,
            showProgress=True
        )

        # ---------------- PDE reference ----------------
        pde_method = PdeSolver(model)

        berm_npv_pde = bermudan_option_npv(
            exercise_times,
            underlying_payoffs,
            pde_method,
            showProgress=True
        )

        # ---------------- BSDE ----------------
        ex_idx = [
            np.where(np.isclose(t_grid, t, atol=1e-12))[0][0]
            for t in exercise_times
        ]

        bsde = BsdeBackwardSolver(
            hw_model=model,
            times=t_grid,
            payoffs=underlying_payoffs,
            exercise_indices=ex_idx,
            payer=False,
            architecture=bsde_hidden,
            activation="relu",
            lr=bsde_lr,
            seed=seed,
        )

        bsde.fit(n_epochs=bsde_epochs, n_paths=bsde_paths)
        berm_npv_bsde = bsde.y0_vals[-1]

        # ---------------- Europeans ----------------
        european_npvs = [s.npv() for s in swaptions]

        if not show_plots:
            return {
                "AMC": berm_npv_amc,
                "PDE": berm_npv_pde,
                "BSDE": berm_npv_bsde,
            }

        # ================== COMBINED FIGURE ==================
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # ----------- Volatility plot (left) -----------
        times = np.linspace(0.0, maturity, 100 * maturity + 1)
        vols  = np.array([model.sigma(t) for t in times])

        ax = axes[0]
        ax.plot(times, vols * 1e4)
        ax.set_xlabel(r"time $t$")
        ax.set_ylabel(r"short rate volatility $\sigma(t)$ (bp)")
        ax.set_title(r"mean reversion $a=%.2f$" % mean_reversion)
        ax.set_ylim((0, 160))

        # ----------- Price comparison (right) ----------
        labels = [f"{e}-{s}" for e, s in zip(expiry_terms, swap_terms)] + [
            "Berm (AMC)", "Berm (PDE)", "Berm (BSDE)"
        ]

        npvs = european_npvs + [
            berm_npv_amc,
            berm_npv_pde,
            berm_npv_bsde
        ]

        x = 4 * np.linspace(1, len(labels), len(labels))

        ax = axes[1]
        ax.bar(x, npvs, 3.0)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=90)
        ax.set_ylabel("option price")
        ax.set_title(
            r"Option values: AMC %.2f | PDE %.2f | BSDE %.2f"
            % (berm_npv_amc, berm_npv_pde, berm_npv_bsde)
        )
        plt.tight_layout()
        plt.savefig(plot_save_path, bbox_inches="tight")
        plt.show()


    def plot_bsde_training_with_table(solvers, npv_amc, filename):

        rows = []

        plt.figure(figsize=(12, 12))

        # -------- top: loss --------
        plt.subplot(2, 1, 1)
        for label, sol in solvers:
            plt.plot(sol.losses, label=label)
        plt.title("Training loss")
        plt.xlabel("epoch")
        plt.ylabel(r"loss (Var[$u_0$])")
        plt.ylim(0, 100)
        plt.grid(True, alpha=0.3)
        plt.legend(ncol=2, fontsize=9)

        # -------- bottom: price --------
        plt.subplot(2, 1, 2)
        for label, sol in solvers:
            plt.plot(sol.y0_vals, label=label)

            price_bsde = sol.y0_vals[-1]
            rel_err_amc = 100.0 * (price_bsde - npv_amc) / npv_amc

            rows.append({
                "Architecture": label.split("|")[-1].strip(),
                "Activation": "softplus",
                "BSDE price": round(price_bsde, 4),
                "Rel. err. AMC [%]": round(rel_err_amc, 2),
            })

        plt.axhline(npv_amc, color="red", linestyle=":", label="AMC")
        plt.title(r"Estimated price $\hat Y_0$")
        plt.xlabel("epoch")
        plt.ylabel("price")
        plt.ylim(550, 1100)
        plt.grid(True, alpha=0.3)
        plt.legend(ncol=2, fontsize=9)

        plt.tight_layout()
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename, dpi=300)
        plt.show()

        df = pd.DataFrame(rows)
        return df
    


    def BermTable(details, exercise_times, berm_price_bsde, npv_amc, npv_pde, elapsed):
        side = "Receiver" if details[0]['call_or_put'] > 0 else "Payer"

        ex = np.array(exercise_times)
        T_max = float(max(d['pay_times'][-1] for d in details))

        rel_amc = (berm_price_bsde - npv_amc) / abs(npv_amc)
        rel_pde = (berm_price_bsde - npv_pde) / abs(npv_pde)

        print("\nBermudan swaption summary")
        print("-" * 70)
        print(f"{'Side':20s}: {side}")
        print(f"{'Exercise dates':20s}: {len(ex)}")
        print(f"{'Exercise window':20s}: [{ex.min():.2f}, {ex.max():.2f}] years")
        print(f"{'Final maturity':20s}: {T_max:.2f} years")
        print(f"{'Runtime':20s}: {elapsed:.2f} s")
        print("-" * 70)
        print(f"{'Method':10s} | {'Price':>12s} | {'Rel. error':>12s}")
        print("-" * 70)
        print(f"{'BSDE':10s} | {berm_price_bsde:12.6f} | {'—':>12s}")
        print(f"{'AMC':10s} | {npv_amc:12.6f} | {rel_amc:12.4f}")
        print(f"{'PDE':10s} | {npv_pde:12.6f} | {rel_pde:12.4f}")
        print("-" * 70)


    def plot_bsde_training(solvers, filename, npv_amc, npv_pde):
        """
        Plot training loss and estimated Y0 for multiple BSDE solvers.

        Parameters
        ----------
        solvers : list of (label, BsdeBackwardSolver)
        filename : str
            Path to save the figure.
        """

        plt.figure(figsize=(12, 12))

        # --- top: loss ---
        plt.subplot(2, 1, 1)
        for label, bsde in solvers:
            plt.plot(bsde.losses, label=label)


        plt.title("Training loss")
        plt.xlabel("epoch")
        plt.ylabel(r"loss (Var[$u_0$])")
        plt.ylim(0,60)
        plt.grid(True, alpha=0.3)
        plt.legend(ncol=2, fontsize=9)

        # --- bottom: price ---
        plt.subplot(2, 1, 2)
        for label, bsde in solvers:
            plt.plot(bsde.y0_vals, label=label)
        plt.axhline(
            npv_amc,
            color="red",
            linestyle=":",
            linewidth=1.5,
            label="AMC"
        )
        plt.axhline(
            npv_pde,
            color="blue",
            linestyle=":",
            linewidth=1.5,
            label="PDE"
        )
        plt.title(r"Estimated price $\hat Y_0$")
        plt.xlabel("epoch")
        plt.ylabel("price")
        plt.ylim(500,1100)
        plt.grid(True, alpha=0.3)
        plt.legend(ncol=2, fontsize=9)

        plt.tight_layout()
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename, dpi=300)
        plt.show()


    @staticmethod
    def plot_y0_paths_with_table(
        results,
        bins,
        xlim,
        ylim,
        title_prefix,
        figsize,
        filename,
    ):
        """
        Plot Y0 distributions for different path counts and return summary table.
        """

        fig, axes = PlotHelpers.plot_y0_snapshots_2x2(
            results=results,
            bins=bins,
            xlim=xlim,
            ylim=ylim,
            figsize=figsize,
            title_prefix=title_prefix,
            filename=filename,
        )

        keys = sorted(results.keys())

        df = pd.DataFrame(
            [
                {
                    "paths": results[k]["paths"],
                    "mean_Y0": results[k]["mean"],
                    "std_Y0":  results[k]["std"],
                }
                for k in keys
            ]
        )

        return df

    

    def plot_bermudan_prices_amc_vs_bsde(
        berm_npv_amc,
        berm_npv_bsde,
        filename,
        figsize=(12, 6),
    ):
        """
        Plot Bermudan option prices (ITM, ATM, OTM) as a function of mean reversion,
        comparing AMC and Deep BSDE results.
        """

        fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)

        # --- AMC ---
        ax = axes[0]
        ax.plot(berm_npv_amc["mean_reversion"], berm_npv_amc["ITM"], label="ITM")
        ax.plot(berm_npv_amc["mean_reversion"], berm_npv_amc["ATM"], label="ATM")
        ax.plot(berm_npv_amc["mean_reversion"], berm_npv_amc["OTM"], label="OTM")
        ax.set_title("American Monte Carlo Regression Prices")
        ax.set_xlabel(r"mean reversion $a$")
        ax.set_ylabel("Bermudan option price")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # --- BSDE ---
        ax = axes[1]
        ax.plot(berm_npv_bsde["mean_reversion"], berm_npv_bsde["ITM"], label="ITM")
        ax.plot(berm_npv_bsde["mean_reversion"], berm_npv_bsde["ATM"], label="ATM")
        ax.plot(berm_npv_bsde["mean_reversion"], berm_npv_bsde["OTM"], label="OTM")
        ax.set_title("Deep BSDE Solver Prices")
        ax.set_xlabel(r"mean reversion $a$")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename, dpi=300)
        plt.show()
