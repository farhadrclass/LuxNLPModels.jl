# julia --project -e 'using Pkg; Pkg.test()'
# GPU:  LUXNLPMODELS_TEST_GPU=true julia --project -e 'using Pkg; Pkg.test()'
using Test, LinearAlgebra, Random
using NLPModels, NLPModelsTest, JSOSolvers, SolverCore, Lux, LuxNLPModels
using DifferentiationInterface: SecondOrder
import ForwardDiff                      # outer backend of the exact Hessian-vector products

const TEST_GPU = get(ENV, "LUXNLPMODELS_TEST_GPU", "false") == "true"
if TEST_GPU
  using CUDA, LuxCUDA
end

# ---------- shared helpers ----------
function toy_data(::Type{T}, rng; n = 64, nin = 3, nout = 2) where {T}
  X = randn(rng, T, nin, n)
  Y = zeros(T, nout, n)
  for i in 1:n
    Y[rand(rng, 1:nout), i] = 1
  end
  return X, Y
end

toy_mlp() = Chain(Dense(3 => 5, tanh), Dense(5 => 2))

toy_nlp(::Type{T}; kwargs...) where {T} =
  LuxNLPModel(toy_mlp(), toy_data(T, Xoshiro(1)); T = T, batchsize = 16, kwargs...)

fd_dir(f, x, v; h = 1e-6) = (f(x + h * v) - f(x - h * v)) / (2h)

const EXACT_HVP = SecondOrder(AutoForwardDiff(), AutoZygote())

@testset "LuxNLPModels" begin
  include("test_minibatch.jl")
  include("test_api.jl")
  include("test_hvp.jl")
  include("test_state.jl")
  include("test_solvers.jl")
  TEST_GPU && include("test_gpu.jl")
end
