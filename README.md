# LuxNLPModels.jl

LuxNLPModels exposes a [Lux.jl](https://github.com/LuxDL/Lux.jl) network and a loss as an
`AbstractNLPModel`, so that [JuliaSmoothOptimizers](https://jso.dev) solvers, first or second
order, can train it on CPU or GPU with minibatches. Where it overlaps with
[FluxNLPModels.jl](https://github.com/JuliaSmoothOptimizers/FluxNLPModels.jl), the API is the same.

## Quick start

```julia
using Lux, LuxNLPModels, JSOSolvers

model = Chain(Dense(784 => 128, relu), Dense(128 => 10))
nlp = LuxNLPModel(model, (Xtrain, Ytrain); batchsize = 128)   # device = gpu_device()

# First order: a new minibatch after every iteration, 1000 iterations.
stats = R2(nlp; max_iter = 1_000, max_time = Inf, atol = 0.0f0, rtol = 0.0f0,
           callback = (nlp, solver, stats) -> minibatch_next_train!(nlp))
```

Data are arrays, or Tuples / NamedTuples of arrays, with observations along the last dimension.
Any loss with Lux's training signature `loss_f(model, ps, st, batch) -> (loss, st, stats)`
works, including Lux's own losses (the default is `CrossEntropyLoss(; logits = Val(true))`).

## Second order: trunk on minibatches

trunk accepts a step by comparing `f` before and after it, so `f`, `∇f` and `Hv` must come from
the same minibatch; switching minibatches in its callback would compare two different functions.
Run a few iterations per minibatch instead, warm-started, with one preallocated solver:

```julia
using SolverCore                                      # GenericExecutionStats
solver, stats = TrunkSolver(nlp), GenericExecutionStats(nlp)
x = copy(nlp.meta.x0)
for k in 1:1_000
  solve!(solver, nlp, stats; x, max_iter = 5, max_time = Inf, atol = 0.0f0, rtol = 0.0f0)
  x .= stats.solution
  minibatch_next_train!(nlp)
  k == 500 && set_batchsize!(nlp, 512)                # larger minibatches as trunk converges
end
```

For a deterministic, full-batch problem (e.g. `lbfgs` or `trunk` to convergence), use
`batchsize = number of observations` and do not switch minibatches.

Complete scripts are in [`examples/`](examples/README.md):

- [`lenet_mnist.jl`](examples/lenet_mnist.jl): LeNet-5 on MNIST, with a CPU/GPU switch, `R2` or
  `trunk`, and a configurable batch size.
- [`pinn_poisson.jl`](examples/pinn_poisson.jl): a small PINN for a 1-D Poisson problem, trained
  with Adam + L-BFGS and with trunk, and compared.
- [`pinn_nernst_planck.jl`](examples/pinn_nernst_planck.jl): the Nernst-Planck PINN optimizer
  benchmark from SciMLBenchmarks / NeuralPDE: Adam, RMSProp, R2, L-BFGS, Adam + L-BFGS and
  trunk on the same loss.

## API

| Function | Purpose |
|---|---|
| `LuxNLPModel(model, data_train; loss_f, batchsize, shuffle, T, device, seed, grad_backend, hvp_backend)` | build the model |
| `obj`, `grad!`, `objgrad!`, `hprod!` (and `hess_op`) | NLPModels API on the current minibatch |
| `minibatch_next_train!(nlp)` | next minibatch; returns `false` when a pass over the data ends |
| `set_batchsize!(nlp, b)` | minibatches of `b` observations from now on; restarts the pass |
| `epoch(nlp)` | number of completed passes |
| `params_view(nlp, x)` | structured Lux parameters at `x`, to call `nlp.model` yourself |

## Behaviour

- Between two minibatch switches (`minibatch_next_train!`, `set_batchsize!`), `obj`, `grad!`
  and `hprod!` describe one function: same minibatch, same Dropout masks, training-mode
  BatchNorm.
- `obj` and `grad!` never update BatchNorm's running statistics. Before evaluating in test mode
  (`Lux.testmode(nlp.st)`), recompute them at `x`, e.g. with training-mode passes
  `_, st, _ = nlp.loss_f(nlp.model, params_view(nlp, x), st, batch)` over training batches.
- `hprod!` is a central difference of two gradients by default: cheap and GPU-friendly, with
  about 1% relative error in `Float32`. For exact products, `using ForwardDiff` and pass
  `hvp_backend = SecondOrder(AutoForwardDiff(), AutoZygote())` (`SecondOrder` is from
  DifferentiationInterface); check that your layers accept ForwardDiff dual numbers, especially
  on GPU.
- On GPU, parameters and the current minibatch live on the device; the data stay on the host and
  one minibatch is copied per switch.

## Development

```bash
julia --project -e 'using Pkg; Pkg.test()'
LUXNLPMODELS_TEST_GPU=true julia --project -e 'using Pkg; Pkg.test()'
```
