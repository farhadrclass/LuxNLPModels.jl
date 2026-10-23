# Examples

```bash
julia --project=examples -e 'using Pkg; Pkg.instantiate()'    # once (Julia 1.10: Pkg.develop(path = ".."); Pkg.instantiate())
```

## PINN for a 1-D Poisson problem: Adam + L-BFGS vs trunk

`pinn_poisson.jl` solves `-u'' = f` on (0, 1) with zero boundary values using a small
physics-informed network, and trains it from the same initial point in two ways:

- Adam, then L-BFGS from its result (the usual PINN recipe);
- trunk, a second-order trust-region method using `hprod!`.

```bash
julia --project=examples examples/pinn_poisson.jl
```

It prints, for each method, the final loss, the relative L2 error against the exact solution,
the number of `obj` / `grad` / `hprod` evaluations and the training time. The budgets
(iterations, learning rate, time cap) are variables at the top of the script.

How the PINN fits the library:

- The boundary conditions are built into the ansatz `u(x) = x(1 - x) N(x)`, so the loss is the
  equation's residual only.
- `u''` is a 3-point finite difference of the network evaluated in one forward pass. With no
  nested AD, gradients and Hessian-vector products work unchanged (Float64, step `δ = 1e-3`).
- All collocation points form one batch (`batchsize = n_points, shuffle = false`), so the
  problem is deterministic, as L-BFGS and trunk expect.

## Nernst-Planck PINN: optimizer benchmark

`pinn_nernst_planck.jl` is adapted from the SciMLBenchmarks "Nernst-Planck Equation
Physics-Informed Neural Network (PINN) Optimizer Benchmarks", itself adapted from *NeuralPDE:
Automating Physics-Informed Neural Networks (PINNs) with Error Approximations* (Zubov et al.,
arXiv:2107.09443), which uses NeuralPDE.jl. It is a 1-D Poisson-Nernst-Planck system for the
potential Φ and the Na⁺ / Cl⁻ concentrations, with three 2→16→16→16→1 `tanh` networks.

The original's equations, constants, conditions, networks and budget (200 iterations; 50 Adam +
150 BFGS) are kept. The differences:

- the loss is written directly as a Lux loss, on fixed grids of collocation points instead of
  adaptive quadrature, with finite-difference derivatives in one forward pass;
- L-BFGS replaces BFGS, which JSOSolvers does not have.

Methods compared:

- Adam and RMSProp, each with step sizes 0.001, 0.005 and 0.05;
- R2;
- L-BFGS;
- Adam + L-BFGS (50 + 150 iterations);
- trunk.

```bash
julia --project=examples examples/pinn_nernst_planck.jl          # CPU
julia --project=examples examples/pinn_nernst_planck.jl --gpu    # GPU, as in the original
```

It prints each method's final loss, time and gradient / Hessian-vector product counts, and
keeps the loss history (time, loss) of every method in `histories`. The end of the script shows
how to plot loss against time, as in the original.

## LeNet-5 on MNIST

`lenet_mnist.jl` loads MNIST, builds LeNet-5, wraps it in a `LuxNLPModel` and trains it on
minibatches, with `R2` (first order) or `trunk` (second order) from JSOSolvers.

```bash
julia --project=examples examples/lenet_mnist.jl                                  # CPU, R2
julia --project=examples examples/lenet_mnist.jl --gpu --batchsize=256 --epochs=5 # GPU if available
julia --project=examples examples/lenet_mnist.jl --solver=trunk --trunk_iters=5   # second order
```

| Setting | Default | |
|---|---|---|
| `--gpu` | off | use the GPU if one works; otherwise a warning and the CPU |
| `--batchsize=N` | 128 | minibatch size |
| `--epochs=N` | 3 | passes over the training data |
| `--solver=NAME` | `R2` | `R2`: next minibatch after every iteration; `trunk`: `trunk_iters` iterations per minibatch |
| `--trunk_iters=N` | 5 | trunk iterations on each minibatch |

The same settings are variables at the top of the script, for running it from the REPL.
Any other JSOSolvers first-order solver takes the same keywords as `R2`: replace it with `fomo`.
