"""
    LuxNLPModels

Expose the training problem of a [Lux.jl](https://lux.csail.mit.edu) model as an
`AbstractNLPModel` (`LuxNLPModel`) or, for least-squares losses, an `AbstractNLSModel`
(`LuxNLSModel`), so that solvers from JuliaSmoothOptimizers (`R2`, `fomo`, `lbfgs`,
`trunk`, ...) can be used to train it and compared against Optimisers.jl rules (`Adam`,
`Descent`, ...) on exactly the same objective.

The optimization variable is the flat vector underlying `ComponentArray(ps)`.
It lives on whatever device the parameters live on (CPU `Vector` or GPU `CuVector`).
"""
module LuxNLPModels

using LinearAlgebra
using Random

using NLPModels
using NLPModels: @lencheck, increment!
using SolverCore: SolverCore, GenericExecutionStats, set_status!, set_iter!, set_time!,
  set_objective!, set_dual_residual!, set_solution!, set_solver_specific!

using LuxCore: LuxCore
using ComponentArrays: ComponentArray, getaxes, getdata

using ADTypes: AbstractADType, AutoZygote, AutoForwardDiff
import DifferentiationInterface as DI
using DifferentiationInterface: Constant

# Loaded so that the default backends (AutoZygote, ForwardDiff-over-Zygote) work out of the box.
using Zygote: Zygote
using ForwardDiff: ForwardDiff

using Optimisers: Optimisers

export LuxNLPModel, LuxNLSModel, ReactantBackend
export current_batch, set_batch!, next_batch!, reset_batches!, use_full_batch!, epoch
export minibatch_next_train!, reset_minibatch_train!
export unflatten, predict, full_objective, full_residual, check_derivatives, work, reset_work!
export TrainingLog, logging_callback, minibatch_callback, budget_callback, combine_callbacks
export train_optimisers, train_by_batches

include("batching.jl")
include("model.jl")
include("nls.jl")
include("common.jl")
include("api.jl")
include("nls_api.jl")
include("callbacks.jl")
include("optimisers.jl")

end # module
