"""
    LuxNLPModels

Expose a Lux network and a loss as an `AbstractNLPModel`, so that JuliaSmoothOptimizers
solvers, first or second order, can train it on CPU or GPU with minibatches.
The API follows FluxNLPModels.jl where the two packages overlap.
"""
module LuxNLPModels

using LinearAlgebra, Random
using NLPModels
using Lux
using ComponentArrays
using DifferentiationInterface          # re-exports AutoZygote, AutoForwardDiff; defines SecondOrder
using MLDataDevices: CPUDevice, cpu_device
using MLUtils: DataLoader, numobs, getobs
import Zygote, ForwardDiff              # load the default AD backends

export AbstractLuxNLPModel, LuxNLPModel
export minibatch_next_train!, reset_minibatch_train!, set_data_train!, epoch
export params_view, testmode_state, accuracy

include("model.jl")
include("minibatch.jl")
include("api.jl")
include("evaluation.jl")

end # module
