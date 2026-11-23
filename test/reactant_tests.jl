# Tests of the Reactant extension (ReactantBackend). Not part of `] test` because Reactant is
# a large optional dependency. Run in an environment with LuxNLPModels, Reactant, Enzyme, Lux,
# ADTypes, NLPModels, JSOSolvers, Test:
#
#   julia --project=<env> test/reactant_tests.jl

using Reactant, Enzyme
using LuxNLPModels, Lux, ADTypes, NLPModels, JSOSolvers, Random, LinearAlgebra, Test
Reactant.set_default_backend("cpu")
include(joinpath(@__DIR__, "..", "examples", "poisson1d_problem.jl"))

@testset "ReactantBackend" begin
  data = poisson_data(; n_interior = 32)
  model = mlp(8, 2)
  ps, st = Lux.setup(Xoshiro(1), model)
  ref = LuxNLSModel(model, residual_taylor, data; ps = f64(ps), st)   # verified DI reference
  x = ref.meta.x0
  v = randn(Xoshiro(2), ref.meta.nvar)
  w = randn(Xoshiro(3), ref.nls_meta.nequ)

  @testset "second-order residual with nested Enzyme (3rd order overall)" begin
    nls = LuxNLSModel(model, residual_enzyme, data; ps = f64(ps), st, backend = ReactantBackend())
    @test nls.fwd_backend isa ReactantBackend
    @test nls.nls_meta.nequ == ref.nls_meta.nequ
    @test residual(nls, x) ≈ residual(ref, x) rtol = 1e-12
    @test jprod_residual(nls, x, v) ≈ jprod_residual(ref, x, v) rtol = 1e-12
    @test jtprod_residual(nls, x, w) ≈ jtprod_residual(ref, x, w) rtol = 1e-12
    f, g = objgrad(nls, x)
    @test f ≈ obj(ref, x) rtol = 1e-12
    @test g ≈ grad(ref, x) rtol = 1e-12
    c = check_derivatives(nls)
    @test c.jprod < 1e-7 && c.adjoint < 1e-12 && c.grad < 1e-12
    @test_throws ArgumentError hprod(nls, x, x)
    stats = R2NLS(nls; subsolver = LSMRSubsolver, max_iter = 20)
    stats_ref = R2NLS(ref; subsolver = LSMRSubsolver, max_iter = 20)
    @test stats.solution ≈ stats_ref.solution rtol = 1e-6
    @test full_objective(nls, x) ≈ obj(ref, x) rtol = 1e-12
  end

  @testset "minibatches reuse the compilation" begin
    nls = LuxNLSModel(model, residual_enzyme, data; ps = f64(ps), st, backend = ReactantBackend(),
                      batchsize = (interior = 8, boundary = nothing), rng = Xoshiro(4))
    refb = LuxNLSModel(model, residual_taylor, data; ps = f64(ps), st,
                       batchsize = (interior = 8, boundary = nothing), rng = Xoshiro(4))
    for _ in 1:3
      @test jtprod_residual(nls, x, ones(10)) ≈ jtprod_residual(refb, x, ones(10)) rtol = 1e-12
      cache = nls.backend_cache
      next_batch!(nls); next_batch!(refb)
      @test nls.backend_cache === cache            # same shapes: compiled functions kept
    end
  end

  @testset "LuxNLPModel" begin
    loss(m, ps, st, b) = sum(abs2, residual_enzyme(m, ps, st, b)) / 2
    nlp = LuxNLPModel(model, loss, data; ps = f64(ps), st, backend = ReactantBackend())
    @test obj(nlp, x) ≈ obj(ref, x) rtol = 1e-12
    @test grad(nlp, x) ≈ grad(ref, x) rtol = 1e-12
  end
end
