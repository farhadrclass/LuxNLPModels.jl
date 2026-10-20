# Examples

Status: written, not executed.

## Setup

```bash
cd examples
julia --project -e 'using Pkg; Pkg.instantiate()'      # Julia >= 1.11 (reads [sources])
julia --project -e 'using Pkg; Pkg.develop(path = ".."); Pkg.instantiate()'   # Julia 1.10
```

## Files

| File | Content |
|---|---|
| `models.jl` | data loaders, validation split, MLP, LeNet-5, small ResNet (BatchNorm), autoencoder |
| `runners.jl` | Adam, JSO solvers in batchwise / callback / fullbatch modes, `run_contract` for contract-following solvers |
| `run.jl` | command line driver |
| `ppo.jl` | PPO with one solve per rollout |
| `solvers/tradam.jl` | TRAdam prototype (leaves this folder when it becomes a package or a JSOSolvers PR) |

## Running

```bash
julia --project=examples examples/run.jl lenet_mnist adam 5 1 --lr=1e-3
julia --project=examples examples/run.jl lenet_mnist tradam 5 1
julia --project=examples examples/run.jl resnet_cifar10 tradam 10 1 --gpu
julia --project=examples examples/run.jl mlp_mnist trunk_full 0 1 --subset=2000 --act=gelu
julia --project=examples examples/run.jl lenet_mnist trunk_full 0 1 --subset=2000 --act=gelu --pool=mean
```

Each run writes one row per epoch (validation metric, oracle counts, forward/backward passes,
solver time) and, separately, the test metric of the final iterate.

## Adding a solver

A solver that follows the stochastic contract, first or second order, plugs into
`run_contract` if `solve_fn(nlp; on_accept!, callback, max_time, atol, rtol, kwargs...)`
returns a `GenericExecutionStats`, calls `on_accept!(nlp)` after each accepted step and then
recomputes f and g, and exposes the iterate as `solver.x`. Add one line to the `runners`
dictionary in `run.jl`.

## Comparing with Adam

- Compare on forward/backward passes (`nforward`, `nbackward`) or wall-clock time, not epochs:
  batchwise modes and contract solvers spend different numbers of passes per epoch, and a
  finite-difference Hessian-vector product costs two gradients.
- Tune every method with the same budget on the validation set; never use the test file for
  tuning or stopping.
- Run several seeds (5-10). The seed fixes initialization and batch order for every method, so
  differences can be analysed as paired; report mean differences with confidence intervals.
- On GPU, pass `--gpu`: timings then synchronize the device before reading the clock.
