"""
    LuxNLPModels

Expose a Lux network and a loss as an `AbstractNLPModel`, so that JuliaSmoothOptimizers
solvers, first or second order, can train it on CPU or GPU with minibatches.
"""
module LuxNLPModels

using LinearAlgebra, Random
using NLPModels
using Lux                               # re-exports cpu_device and gpu_device
using ComponentArrays
using DifferentiationInterface          # re-exports AutoZygote
using MLUtils: DataLoader, numobs
import Zygote                           # load the default AD backend

export LuxNLPModel, minibatch_next_train!, set_batchsize!, epoch, params_view

include("model.jl")
include("api.jl")
include("minibatch.jl")

end # module
