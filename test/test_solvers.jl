@testset "JSO solvers" begin
  full() = toy_nlp(Float64; batchsize = 64, shuffle = false)    # one batch: a deterministic NLP

  @testset "first order on the full batch: $solve" for solve in (fomo, R2, lbfgs)
    nlp = full()
    f0 = obj(nlp, nlp.meta.x0)
    stats = solve(nlp; max_iter = 50, max_time = 60.0)
    @test stats.objective < f0
  end

  @testset "second order on the full batch: trunk" begin
    nlp = full()
    f0 = obj(nlp, nlp.meta.x0)
    stats = trunk(nlp; max_iter = 20, max_time = 60.0)
    @test stats.objective < f0
    @test neval_hprod(nlp) > 0
  end

  @testset "minibatches switched in a callback (FluxNLPModels pattern)" begin
    nlp = toy_nlp(Float64)
    cb(nlp, solver, stats) = minibatch_next_train!(nlp)
    stats = fomo(nlp; max_iter = 40, max_time = 60.0, callback = cb)
    @test isfinite(stats.objective)
    @test epoch(nlp) >= 1
  end
end
