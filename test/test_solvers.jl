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

  @testset "trunk on minibatches: warm-started solves, one solver" begin
    nlp = toy_nlp(Float64)
    solver, stats = TrunkSolver(nlp), GenericExecutionStats(nlp)
    x = copy(nlp.meta.x0)
    for k in 1:8                                       # 2 passes of 4 minibatches
      k == 5 && set_batchsize!(nlp, 32)                # grow the batch halfway
      solve!(solver, nlp, stats; x, max_iter = 3, max_time = 60.0)
      x .= stats.solution
      minibatch_next_train!(nlp)
    end
    @test neval_hprod(nlp) > 0
    loss_all = full()                                  # same initial point and data
    @test obj(loss_all, x) < obj(loss_all, loss_all.meta.x0)
  end

  @testset "trunk with an exact hvp_backend" begin
    nlp = toy_nlp(Float64; batchsize = 64, shuffle = false, hvp_backend = EXACT_HVP)
    f0 = obj(nlp, nlp.meta.x0)
    stats = trunk(nlp; max_iter = 20, max_time = 60.0)
    @test stats.objective < f0
  end

  @testset "minibatches switched in a callback (FluxNLPModels pattern)" begin
    nlp = toy_nlp(Float64)
    cb(nlp, solver, stats) = minibatch_next_train!(nlp)
    stats = fomo(nlp; max_iter = 40, max_time = 60.0, callback = cb)
    @test isfinite(stats.objective)
    @test epoch(nlp) >= 1
  end
end
