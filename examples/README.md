# Examples

```bash
cd examples
julia --project -e 'using Pkg; Pkg.instantiate()'                                 # Julia >= 1.11
julia --project -e 'using Pkg; Pkg.develop(path = ".."); Pkg.instantiate()'       # Julia 1.10
```

| File | Content |
|---|---|
| `models.jl` | data loaders, validation split, MLP, LeNet-5, small ResNet (BatchNorm), autoencoder |
| `instrumentation.jl` | `PassCounter` (forward/backward passes, not just oracle calls), `mean_loss` |
| `runners.jl` | Adam; JSO solvers in batchwise, callback and fullbatch modes; `run_contract` |
| `run.jl` | command line driver |
| `rl.jl`, `ppo.jl` | PPO objective; one solve per rollout with `set_data_train!` |
| `solvers/tradam.jl` | TRAdam prototype; leaves this folder when it becomes a package or a PR |

## Running

```bash
julia --project=examples examples/run.jl lenet_mnist adam 5 1 --lr=1e-3
julia --project=examples examples/run.jl lenet_mnist tradam 5 1
julia --project=examples examples/run.jl resnet_cifar10 tradam 10 1 --gpu
julia --project=examples examples/run.jl mlp_mnist trunk_full 0 1 --subset=2000 --act=gelu
julia --project=examples examples/run.jl lenet_mnist trunk_full 0 1 --subset=2000 --act=gelu --pool=mean
```

Each run writes one row per epoch (validation metric, oracle calls, forward/backward passes,
solver time) and, separately, the test metric of the final iterate.

## Adding a solver

A solver, first or second order, plugs into `run_contract` if
`solve_fn(nlp; on_accept!, callback, max_time, atol, rtol, kwargs...)` returns a
`GenericExecutionStats`, calls `on_accept!(nlp)` after each accepted step and then recomputes
f and g, and exposes the iterate as `solver.x`. Add one line to the `runners` dictionary in
`run.jl`.

## Comparing with Adam

- Compare on forward/backward passes or wall-clock time, not epochs.
- Tune every method with the same budget on the validation split; use the test file only once.
- Run 5-10 seeds. The seed fixes initialization and minibatch order for every method, so
  differences are paired; report mean differences with confidence intervals.
- With `--gpu`, timings synchronize the device before reading the clock.
- Smooth-function analyses (second order, negative curvature) need `--act=gelu --pool=mean`:
  ReLU and max pooling are only piecewise smooth.
