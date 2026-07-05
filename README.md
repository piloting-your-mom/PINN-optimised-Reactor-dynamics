# PINN-Optimised Reactor Dynamics

A Physics-Informed Neural Network (PINN) framework combined with CasADi for modelling and dynamic optimization of a non-isothermal semi-batch reactor with series reactions and heat removal constraints.

## Overview

This project implements an advanced approach to chemical reactor optimization by:

1. **Training a PINN Surrogate Model** - A physics-informed neural network that learns to approximate the differential equations governing reactor dynamics
2. **Dynamic Optimization** - Using CasADi with Ipopt to solve the optimal control problem and maximize product yield
3. **Validation** - Fitting polynomial approximations to optimal controls and validating results using numerical integration

## Problem Statement

### Reactor System
- **Type**: Non-isothermal semi-batch reactor
- **Reactions**: Series reactions (A + B → C → products)
- **Variables**: 
  - Concentrations: cA, cB, cC (mol/L)
  - Reactor volume: V (L)
  - Control inputs: Feed rate u (L/h), Temperature T (°C)

### Constraints
- Temperature bounds: 20°C ≤ T ≤ 50°C
- Feed rate bounds: 0.0 ≤ u ≤ 1.0 L/h
- Volume constraint: 1.0 ≤ V ≤ 1.1 L
- Heat generation limit: q_rx ≤ 150 kW
- Rate-of-change limits: ΔT ≤ 1.0°C/step, Δu ≤ 0.01 L/h/step

### Objective
**Maximize**: Total moles of product C at final time → J = cC(tf) × V(tf)

## Methodology

### 1. PINN Surrogate Training

```python
# Neural network architecture
Input layer: 6 features [cA, cB, cC, V, u, T]
Hidden layers: 64 neurons with Tanh activation
Output layer: 4 features [dcA/dt, dcB/dt, dcC/dt, dV/dt]
```

- **Training data**: 20,000 randomized state samples
- **Loss function**: Mean Squared Error (MSE)
- **Optimizer**: Adam (lr=0.001)
- **Epochs**: 1,000

The PINN learns to approximate the true ODE dynamics:

$$\frac{dcA}{dt} = -k_1 c_A c_B - \frac{u}{V} c_A$$

$$\frac{dcB}{dt} = -k_1 c_A c_B + \frac{u}{V} (c_{B,in} - c_B)$$

$$\frac{dcC}{dt} = k_1 c_A c_B - k_2 c_C - \frac{u}{V} c_C$$

$$\frac{dV}{dt} = u$$

where reaction rate constants follow Arrhenius temperature dependence:
$$k_i = k_{i,0} \exp\left(-\frac{E_i}{RT}\right)$$

### 2. Dynamic Optimization with CasADi

- **Decision variables**: Temperature profile T(t) and feed rate u(t) over N=200 control intervals
- **Transcription method**: Multiple shooting with RK4 numerical integration
- **Constraints**: State, control, and rate-of-change constraints
- **Solver**: Ipopt (interior-point method)

### 3. Validation

Polynomial fitting is applied to optimal u(t) and T(t), then validated using `scipy.integrate.solve_ivp`.

## Installation

```bash
# Install dependencies
pip install casadi torch numpy scipy matplotlib

# For Google Colab (included in notebook)
!pip install casadi
```

## Key Parameters

| Parameter | Value | Unit |
|-----------|-------|------|
| k₁₀ (pre-exponential factor) | 4.0 | - |
| k₂₀ (pre-exponential factor) | 800.0 | - |
| E₁ (activation energy) | 6,000 | J/mol |
| E₂ (activation energy) | 20,000 | J/mol |
| ΔH₁ (heat of reaction 1) | -30,000 | J/mol |
| ΔH₂ (heat of reaction 2) | -10,000 | J/mol |
| Gas constant R | 8.314 | J/(mol·K) |
| Feed inlet conc. (cBin) | 20.0 | mol/L |
| Final time tf | 0.5 | h |

## Results

The optimization yields:

```
Exit Status: Optimal Solution Found
Iterations: 27
Final Objective: J = cC(tf) × V(tf) = 2.061 mol
Solver Time: 9.12 seconds
```

### Output Plots
1. **Concentrations**: Time evolution of reactant A, intermediate B, and product C
2. **Temperature Profile**: Optimal temperature trajectory with bounds
3. **Feed Rate Profile**: Optimal feed rate control input
4. **Reactor Volume**: Volume trajectory during operation

## Files

- `CL299_proj (2).ipynb` - Main Jupyter notebook with complete implementation

## Usage

Run cells sequentially in the Jupyter notebook:

1. Install CasADi and PyTorch
2. Define system parameters
3. Create and train PINN model
4. Build CasADi NLP problem
5. Solve optimization problem
6. Visualize results

## Dependencies

- **PyTorch**: Neural network training
- **CasADi**: Symbolic optimization
- **Ipopt**: Nonlinear programming solver
- **NumPy/SciPy**: Scientific computing
- **Matplotlib**: Visualization

## License

MIT License - See LICENSE file for details

## References

- CasADi: Andersson, J.A.E., et al. (2019). "CasADi: a software framework for nonlinear optimization and optimal control"
- PINNs: Raissi, M., Perdikaris, P., & Karniadakis, G.E. (2019). "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems"

---

**Author**: piloting-your-mom  
**Last Updated**: 2026-04-5  
**Status**: Compelete
