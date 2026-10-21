"""
    LuxNLPModels

Expose a Lux network and a loss as an `AbstractNLPModel`, so that JuliaSmoothOptimizers
solvers, first or second order, can train it on CPU or GPU with minibatches.
"""
module LuxNLPModels

using LinearAlgebra, Random
using NLPModels
using Lux
using ComponentArrays
using DifferentiationInterface          # re-exports AutoZygote
using MLDataDevices: cpu_device
using MLUtils: DataLoader, numobs, getobs
import Zygote                           # load the default AD backend

export LuxNLPModel, minibatch_next_train!, epoch, params_view, accuracy

include("model.jl")
include("api.jl")
include("minibatch.jl")
include("evaluation.jl")

end # module
