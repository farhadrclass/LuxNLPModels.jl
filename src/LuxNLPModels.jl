"""
    LuxNLPModels

Expose a Lux network and a training objective as an `AbstractNLPModel`, so that
JuliaSmoothOptimizers solvers, first or second order, can train it on CPU or GPU.

Design:
- variables = flattened Lux parameters, viewed through ComponentArray axes, stored on `device`;
- objective = Lux training signature `(model, ps, st, batch) -> (loss, st, stats)`;
- data = an `AbstractBatchSource`; batches have a fixed size and are gathered into
  preallocated host and device buffers;
- `obj`, `grad!`, `objgrad!` and `hprod!` are all evaluated on the current batch, which only
  the solver changes (`next_batch!`); see `stochastic_interface.jl` for the contract;
- every evaluation starts from the committed layer state; `commit_state!` adopts the state
  returned by the last `obj` call; stochastic layers are reseeded at each batch switch;
- derivatives via DifferentiationInterface with prepared operators, reused across batches;
- `pass_counts(nlp)` records forward and backward passes, for cost comparisons.

Solvers are not part of this package.

STATUS: written, not executed. Lines marked CHECK depend on package behaviour to confirm.
"""
module LuxNLPModels

using LinearAlgebra, Random, Statistics
using NLPModels
using Lux
using NNlib: logsoftmax
using ComponentArrays
using DifferentiationInterface          # re-exports AutoZygote, AutoForwardDiff; defines SecondOrder
using MLDataDevices: cpu_device
import Zygote, ForwardDiff              # load the default AD backends

export LuxNLPModel
export AbstractBatchSource, ArrayBatchSource
export next_batch!, set_batch!, use_full_batch!, replace_data!, current_batch, epoch, commit_state!
export params_view, pass_counts, set_hvp_backend!
export recalibrate_state!, mean_objective, accuracy
export classification_objective, mse_objective, ppo_objective

include("batch_sources.jl")
include("objectives.jl")
include("model.jl")
include("stochastic_interface.jl")
include("api.jl")
include("hvp.jl")
include("evaluation.jl")
include("rl.jl")

end # module
