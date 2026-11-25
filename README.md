# LuxNLPModels

This package provides an interface between neural networks defined with
[Lux.jl](https://lux.csail.mit.edu) and the
[NLPModels](https://github.com/JuliaSmoothOptimizers/NLPModels.jl) API, so that solvers such as those of
[JSOSolvers](https://github.com/JuliaSmoothOptimizers/JSOSolvers.jl) can train them.
The variables are the flattened parameters of the network.

- `LuxNLPModel` evaluates a loss `loss_f(ŷ, y)` on the current training minibatch:
  `obj`, `grad!`, `objgrad!`, `hprod!`, `hess_structure!`, `hess_coord!`.
- `LuxNLSModel` evaluates residuals `residual_f(ŷ, y)` on the current training minibatch, with
  objective `½‖F‖²`: `residual!`, `jprod_residual!`, `jtprod_residual!`, `jac_structure_residual!`,
  `jac_coord_residual!`.

The explicit Hessian and Jacobian are dense; prefer the products for anything but small networks.

## Minibatches

As in [FluxNLPModels](https://github.com/JuliaSmoothOptimizers/FluxNLPModels.jl), the training and
test data are iterators over minibatches, e.g. `MLUtils.DataLoader`s, and `minibatch_next_train!(nlp)`
selects the next training minibatch; it returns `false` at the end of an epoch.
Use `[(x, y)]` for the full batch.

```julia
using LuxNLPModels, Lux, JSOSolvers, MLUtils

model = Chain(Dense(2 => 16, tanh), Dense(16 => 1))
x = randn(Float32, 2, 1000)
y = sin.(x[1:1, :])

nlp = LuxNLPModel(model, DataLoader((x, y); batchsize = 100, shuffle = true))
stats = R2(nlp; callback = (nlp, solver, stats) -> minibatch_next_train!(nlp), max_iter = 100)

nls = LuxNLSModel(model, [(x, y)])
stats = R2NLS(nls; subsolver = LSMRSubsolver)
```

A callback that changes the minibatch changes the objective under the solver, which still holds
values from the previous minibatch.
`example/ode_residual.jl` shows the update R2NLS needs.
R2 also keeps the objective and the gradient norm in local variables, so with the callback above
its ratio ρ compares two consecutive minibatches.

Pass `device = gpu_device()` to the constructor to work on the GPU.

## Other data

The objective and the residual are computed by `local_loss(nlp, ps, batch)` and
`local_residual(nlp, ps, batch)`, where `ps` are the network parameters.
To use data that are not of the form `(x, y)`, add a method for your batch type;
see `example/ode_residual.jl`.

## Examples

- `example/MNIST_mlp.jl`: a multilayer perceptron on MNIST, trained with R2 and with Adam.
- `example/ode_residual.jl`: a network fitted to the solution of an ODE through its residuals, with
  R2NLS on the full batch, and with R2NLS and Adam on minibatches.
