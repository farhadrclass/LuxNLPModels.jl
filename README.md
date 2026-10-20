# LuxNLPModels.jl

LuxNLPModels exposes a [Lux.jl](https://github.com/LuxDL/Lux.jl) network and a loss as an
`AbstractNLPModel`, so that [JuliaSmoothOptimizers](https://jso.dev) solvers, first or second
order, can train it on CPU or GPU with minibatches. Where it overlaps with
[FluxNLPModels.jl](https://github.com/JuliaSmoothOptimizers/FluxNLPModels.jl), the API is the same.

## Quick start

```julia
using Lux, LuxNLPModels, JSOSolvers
using MLDataDevices: gpu_device

model = Chain(Dense(784 => 128, gelu), Dense(128 => 10))
nlp = LuxNLPModel(model, (Xtrain, Ytrain), (Xval, Yval); batchsize = 128, device = gpu_device())

# Full-batch, deterministic: build the model with batchsize = number of observations.
# Minibatches with an existing solver: switch the minibatch in the callback.
stats = fomo(nlp; max_iter = 1_000, callback = (nlp, solver, stats) -> minibatch_next_train!(nlp))
accuracy(nlp, stats.solution)
```

Data are arrays, or Tuples / NamedTuples of arrays, with observations along the last
dimension; `Y` is one-hot for `accuracy`. Any loss with Lux's training signature
`loss_f(model, ps, st, batch) -> (loss, st, stats)` works, including Lux's own losses.

## API

| Function | Purpose |
|---|---|
| `LuxNLPModel(model, data_train, data_test; loss_f, batchsize, shuffle, T, device, seed, grad_backend, hvp_backend)` | build the model |
| `obj`, `grad!`, `objgrad!`, `hprod!` | NLPModels API on the current minibatch |
| `minibatch_next_train!(nlp)` | next minibatch; returns `false` when a pass ends |
| `reset_minibatch_train!(nlp)` | restart the current pass |
| `set_data_train!(nlp, data)` | replace the training data (e.g. a new RL rollout) |
| `epoch(nlp)` | completed passes |
| `params_view(nlp, x)` | structured parameters at `x` (a view) |
| `testmode_state(nlp, x)` | layer state for test-mode evaluation at `x` |
| `accuracy(nlp, x; data)` | test-mode classification accuracy |

## What a solver can rely on

Between two minibatch switches, `obj`, `grad!`, `objgrad!` and `hprod!` describe one function:
same minibatch, same layer state, same Dropout masks, training-mode normalization. Switching
the minibatch is the only operation that changes that function. A solver that compares f at two
points (a ratio test, a line search) therefore compares values of one function as long as it
does not switch the minibatch in between.

Evaluation (`testmode_state`, `accuracy`) never changes the minibatch or the layer state used by
the solver, so it can run inside a callback.

## Differences from FluxNLPModels

| | FluxNLPModels | LuxNLPModels |
|---|---|---|
| BatchNorm and Dropout | Flux layers are active only inside a gradient call, so `obj` evaluates the test-mode network and `grad!` the training-mode one | `obj`, `grad!` and `hprod!` evaluate the same training-mode function |
| Dropout masks | redrawn at every evaluation inside a gradient | fixed between minibatch switches |
| BatchNorm statistics at evaluation | whatever the layers accumulated | recomputed at `x` by `testmode_state`, independently of the solver |
| `accuracy` | at the weights of the last `obj`/`grad!` call | at an explicit `x` |
| Second order | — | `hprod!`: forward-over-reverse, or finite differences on GPU |
| `objgrad!` | a separate forward pass for f, then a forward and backward pass for g | one forward and one backward pass |
| GPU minibatches | device passed per call; the batch loaded at the end of a pass is not moved | device fixed at construction; batches copied into a preallocated device buffer |
| Precision | converted on the fly (Float16, Float32, Float64) | fixed at construction (Float32 or Float64); a vector of another type is rejected |

The FluxNLPModels column is read from its source (`main` branch), not from running it.

## GPU

Parameters, workspaces and the current minibatch live on `device`; data stay on the host and one
minibatch is copied per switch. The default `hvp_backend` on GPU is `:finite_difference`
(two gradients per product). In Float32, ratio tests lose meaning when the predicted decrease is
below about `1e-7 |f|`.

## Examples

`examples/` holds the model zoo (MLP, LeNet-5, ResNet with BatchNorm, autoencoder), runners for
Adam and JSO solvers, a pass-counting wrapper, PPO, and the TRAdam solver prototype. See
`examples/README.md`.

## Development

```bash
julia --project -e 'using Pkg; Pkg.test()'
LUXNLPMODELS_TEST_GPU=true julia --project -e 'using Pkg; Pkg.test()'
```
