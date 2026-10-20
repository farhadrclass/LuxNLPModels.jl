@testset "JSO solvers on a fixed batch" begin
  X, Y = toy_data(Float64, Xoshiro(8))
  make() = LuxNLPModel(toy_mlp(), ArrayBatchSource((X, Y); batchsize = 64, shuffle = false); T = Float64)

  @testset "first order: $solve" for solve in (fomo, R2, lbfgs)
    nlp = make()
    f0 = obj(nlp, nlp.meta.x0)
    stats = solve(nlp; max_iter = 50, max_time = 60.0)
    @test stats.objective < f0
  end

  @testset "second order: trunk (Newton-CG through hprod)" begin
    nlp = make()
    f0 = obj(nlp, nlp.meta.x0)
    stats = trunk(nlp; max_iter = 20, max_time = 60.0)
    @test stats.objective < f0
    @test neval_hprod(nlp) > 0
  end
end
