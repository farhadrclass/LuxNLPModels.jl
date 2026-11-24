module LuxNLPModels

using LinearAlgebra, Random
using ComponentArrays, ForwardDiff, Lux, NLPModels, Zygote

export AbstractLuxNLPModel, AbstractLuxNLSModel, LuxNLPModel, LuxNLSModel
export local_loss, local_residual
export reset_minibatch_train!, reset_minibatch_test!
export minibatch_next_train!, minibatch_next_test!
export accuracy

abstract type AbstractLuxNLPModel{T, S} <: AbstractNLPModel{T, S} end
abstract type AbstractLuxNLSModel{T, S} <: AbstractNLSModel{T, S} end

"""
    LuxNLPModel{T, S} <: AbstractLuxNLPModel{T, S}

Data structure that makes the interface between neural networks defined with
[Lux.jl](https://lux.csail.mit.edu) and
[NLPModels](https://github.com/JuliaSmoothOptimizers/NLPModels.jl).
The variables are the flattened parameters of the network.
A LuxNLPModel has fields

- `meta` and `counters` retain information about the `LuxNLPModel`;
- `model` is the Lux network, `ps_axes` the structure of its parameters and `st` its state;
- `loss_f` is the loss function, called as `loss_f(ŷ, y)`;
- `device` moves data to the device of the parameters;
- `training_minibatch_iterator` and `test_minibatch_iterator` iterate over minibatches;
- `current_training_minibatch` is the minibatch used to evaluate the objective;
- `current_test_minibatch` is the current test minibatch;
- `current_training_minibatch_status` and `current_test_minibatch_status` are the iterator states.
"""
mutable struct LuxNLPModel{T, S, M, F, A, ST, D} <: AbstractLuxNLPModel{T, S}
  meta::NLPModelMeta{T, S}
  counters::Counters
  model::M
  loss_f::F
  ps_axes::A
  st::ST
  device::D
  training_minibatch_iterator
  test_minibatch_iterator
  current_training_minibatch
  current_test_minibatch
  current_training_minibatch_status
  current_test_minibatch_status
end

"""
    LuxNLPModel(model, data_train, data_test = data_train; loss_f = MSELoss(), rng = Random.default_rng(), ps, st, device = cpu_device())

Build a `LuxNLPModel` from the Lux network `model`.
`data_train` and `data_test` are iterators over minibatches `(x, y)`, e.g. `MLUtils.DataLoader`s;
use `[(x, y)]` for the full batch.
`ps` and `st` default to `Lux.initialparameters(rng, model)` and `Lux.initialstates(rng, model)`;
the element type of `ps` is the element type of the problem.
The state `st` is not updated by evaluations.
"""
function LuxNLPModel(
  model,
  data_train,
  data_test = data_train;
  loss_f = MSELoss(),
  rng::AbstractRNG = Random.default_rng(),
  ps = Lux.initialparameters(rng, model),
  st = Lux.initialstates(rng, model),
  device = cpu_device(),
)
  x0, ps_axes, st, train, test = lux_setup(model, ps, st, device, data_train, data_test)
  meta = NLPModelMeta(length(x0); x0 = x0, variable_bounds_analysis = false)
  return LuxNLPModel(
    meta,
    Counters(),
    model,
    loss_f,
    ps_axes,
    st,
    device,
    data_train,
    data_test,
    train[1],
    test[1],
    train[2],
    test[2],
  )
end

"""
    LuxNLSModel{T, S} <: AbstractLuxNLSModel{T, S}

Nonlinear least-squares counterpart of [`LuxNLPModel`](@ref): the objective is `½‖F(w)‖²`,
where the residual `F` is evaluated on the current training minibatch.
It has the fields of a `LuxNLPModel`, with `nls_meta` and `residual_f`, called as
`residual_f(ŷ, y)`, instead of `loss_f`.
"""
mutable struct LuxNLSModel{T, S, M, F, A, ST, D} <: AbstractLuxNLSModel{T, S}
  meta::NLPModelMeta{T, S}
  nls_meta::NLSMeta{T, S}
  counters::NLSCounters
  model::M
  residual_f::F
  ps_axes::A
  st::ST
  device::D
  training_minibatch_iterator
  test_minibatch_iterator
  current_training_minibatch
  current_test_minibatch
  current_training_minibatch_status
  current_test_minibatch_status
end

"""
    LuxNLSModel(model, data_train, data_test = data_train; residual_f = (ŷ, y) -> ŷ .- y, rng = Random.default_rng(), ps, st, device = cpu_device())

Build a `LuxNLSModel` from the Lux network `model`; see [`LuxNLPModel`](@ref) for the arguments.
The number of residuals is set by the first training minibatch, so all minibatches must
produce the same number of residuals.
"""
function LuxNLSModel(
  model,
  data_train,
  data_test = data_train;
  residual_f = (ŷ, y) -> ŷ .- y,
  rng::AbstractRNG = Random.default_rng(),
  ps = Lux.initialparameters(rng, model),
  st = Lux.initialstates(rng, model),
  device = cpu_device(),
)
  x0, ps_axes, st, train, test = lux_setup(model, ps, st, device, data_train, data_test)
  n = length(x0)
  meta = NLPModelMeta(n; x0 = x0, variable_bounds_analysis = false)
  nls = LuxNLSModel(
    meta,
    NLSMeta(1, n; x0 = x0),
    NLSCounters(),
    model,
    residual_f,
    ps_axes,
    st,
    device,
    data_train,
    data_test,
    train[1],
    test[1],
    train[2],
    test[2],
  )
  nequ = length(residual_w(nls, x0, nls.current_training_minibatch))
  nls.nls_meta = NLSMeta(nequ, n; x0 = x0, nnzj = nequ * n)
  return nls
end

function lux_setup(model, ps, st, device, data_train, data_test)
  ps = device(ComponentArray(ps))
  train, test = iterate(data_train), iterate(data_test)
  (train === nothing || test === nothing) && error("train data or test data is empty")
  train = (device(train[1]), train[2])
  test = (device(test[1]), test[2])
  return copy(getdata(ps)), getaxes(ps), device(st), train, test
end

include("LuxNLPModels_methods.jl")
include("LuxNLSModels_methods.jl")
include("utils.jl")

end
