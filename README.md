## Overview

This repository implements a deep BSDE backward solver for pricing Bermudan swaptions in the Hull-White model. The solver is based on a time-discretized forward-backward stochastic differential equation (FBSDE) formulation and is implemented using PyTorch.

## Structure

- **Deep BSDE Solver**  
  Implements the backward solver for Bermudan swaptions as described in the corresponding algorithm. The solver approximates the value process and control process using neural networks.

- **Hull-White Model**  
  The interest rate dynamics follow the one-factor Hull-White model. The implementation supports time-dependent volatility structures and Monte Carlo simulation of short-rate paths.

- **Monte Carlo Simulation**  
  The simulation of interest rate paths is based on the framework provided in:
  https://github.com/sschlenkrich/InterestRateModelling_examples

- **Swaption Setup**  
  The construction of the underlying swaps, the Hull-White model specification, and the intrinsic payoff functions are adapted from the same repository as above.

## Implementation Details

- The solver is implemented as a **PyTorch module**.
- Time discretization is performed on a fixed grid including all Bermudan exercise dates.
- Conditional expectations are approximated via neural networks.
- Training is performed using stochastic gradient descent over simulated sample paths.

## Experiments

All numerical experiments are provided in the accompanying Python notebook. These experiments reproduce the results presented in the thesis and allow for further exploration of model parameters and network architectures.

## References

The implementation follows the theoretical framework developed in the thesis:
- BSDE foundations and numerical schemes
- Deep BSDE solvers
- Hull-White interest rate model

Parts of the numerical setup and analysis are aligned with existing examples to ensure comparability.
