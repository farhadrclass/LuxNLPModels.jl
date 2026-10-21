# LuxNLPModels.jl

LuxNLPModels exposes a [Lux.jl](https://github.com/LuxDL/Lux.jl) network and a loss as an
`AbstractNLPModel`, so that [JuliaSmoothOptimizers](https://jso.dev) solvers, first or second
order, can train it on CPU or GPU with minibatches. Where it overlaps with
[FluxNLPModels.jl](https://github.com/JuliaSmoothOptimizers/FluxNLPModels.jl), the API is the same.

## Quick start

```julia
using Lux, LuxNLPModels, JSOSolvers

model = Chain(Dense(784 => 128, relu), Dense(128 => 10))
nlp = LuxNLPModel(model, (Xtrain, Ytrain), (Xtest, Ytest); batchsize = 128)   # device = gpu_device()

# A new minibatch after every iteration, 1000 iterations.
stats = R2(nlp; max_iter = 1_000, max_time = Inf, atol = 0.0f0, rtol = 0.0f0,
           callback = (nlp, solver, stats) -> minibatch_next_train!(nlp))
accuracy(nlp, stats.solution)
```

Data are arrays, or Tuples / NamedTuples of arrays, with observations along the last dimension;
`Y` is one-hot for `accuracy`. Any loss with Lux's training signature
`loss_f(model, ps, st, batch) -> (loss, st, stats)` works, including Lux's own losses (the
default is `CrossEntropyLoss(; logits = Val(true))`).

A complete script, LeNet-5 on MNIST with a CPU/GPU switch and a configurable batch size, is in
[`examples/lenet_mnist.jl`](examples/lenet_mnist.jl).

## API

| Function | Purpose |
|---|---|
| `LuxNLPModel(model, data_train, data_test; loss_f, batchsize, shuffle, T, device, seed)` | build the model |
| `obj`, `grad!`, `objgrad!`, `hprod!` | NLPModels API on the current minibatch |
| `minibatch_next_train!(nlp)` | next minibatch; returns `false` when a pass over the data ends |
| `epoch(nlp)` | number of completed passes |
| `accuracy(nlp, x; data = test data)` | test-mode classification accuracy at `x` |
| `params_view(nlp, x)` | structured Lux parameters at `x`, to call `nlp.model` yourself |

For a deterministic, full-batch problem (e.g. `lbfgs` or `trunk`), use
`batchsize = number of observations` and do not switch minibatches.

## Behaviour

- Between two calls to `minibatch_next_train!`, `obj`, `grad!` and `hprod!` describe one
  function: same minibatch, same Dropout masks, training-mode BatchNorm.
- `accuracy` runs in test mode, recomputing BatchNorm statistics at `x`; it never changes the
  minibatch or the counters, so it can be called inside a callback.
- `hprod!` is a central difference of two gradients.
- On GPU, parameters and the current minibatch live on the device; the data stay on the host and
  one minibatch is copied per switch.

## Development

```bash
julia --project -e 'using Pkg; Pkg.test()'
LUXNLPMODELS_TEST_GPU=true julia --project -e 'using Pkg; Pkg.test()'
```
