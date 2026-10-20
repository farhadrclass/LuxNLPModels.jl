@testset "NLPModels API" begin
  @testset "metadata and dimension checks" begin
    nlp = toy_nlp(Float64)
    @test nlp.meta.nnzh == 0
    @test !nlp.meta.hess_available
    @test nlp.meta.hprod_available
    # No Hessian matrix is provided, only products.
    check_nlp_dimensions(nlp; exclude = [hess, hess_coord, ghjvprod, jth_hess, jth_hess_coord, jth_hprod])
  end

  @testset "precision $T" for T in (Float32, Float64)
    nlp = toy_nlp(T)
    x = nlp.meta.x0
    @test eltype(x) == T
    @test obj(nlp, x) isa T
    @test eltype(grad(nlp, x)) == T
    @test eltype(hprod(nlp, x, ones(T, length(x)))) == T
    X, Y = toy_data(T, Xoshiro(1))
    @test_throws ArgumentError LuxNLPModel(toy_mlp(), ArrayBatchSource((Float16.(X), Float16.(Y)); batchsize = 16); T = T)
  end

  @testset "gradients; prepared AD reused across batches and re-prepared on size changes" begin
    nlp = toy_nlp(Float64)
    x = copy(nlp.meta.x0)
    v = randn(Xoshiro(2), length(x))
    f = z -> obj(nlp, z)
    for _ in 1:3                                     # same preparation, three different batches
      g = grad(nlp, x)
      @test dot(g, v) ≈ fd_dir(f, x, v) rtol = 1e-6
      fx, g2 = objgrad!(nlp, x, similar(x))
      @test fx ≈ obj(nlp, x)
      @test g2 ≈ g
      next_batch!(nlp)
    end
    use_full_batch!(nlp)                             # 64 observations: re-prepared
    @test size(current_batch(nlp)[1], 2) == 64
    @test dot(grad(nlp, x), v) ≈ fd_dir(f, x, v) rtol = 1e-6
    @test obj(nlp, x) ≈ mean_objective(nlp, x) rtol = 1e-12
    next_batch!(nlp)                                 # back to the nominal size: re-prepared
    @test size(current_batch(nlp)[1], 2) == 16
    @test dot(grad(nlp, x), v) ≈ fd_dir(f, x, v) rtol = 1e-6
  end

  @testset "counters and pass counts" begin
    nlp = toy_nlp(Float64; hvp_backend = :finite_difference)
    x = nlp.meta.x0
    NLPModels.reset!(nlp)
    objgrad!(nlp, x, similar(x))
    @test neval_obj(nlp) == 1 && neval_grad(nlp) == 1
    @test pass_counts(nlp).forward == 1 && pass_counts(nlp).backward == 1
    obj(nlp, x)
    hprod(nlp, x, ones(length(x)))                   # finite differences: 2 forward + 2 backward
    @test pass_counts(nlp).forward == 4 && pass_counts(nlp).backward == 3
    @test neval_hprod(nlp) == 1
    NLPModels.reset!(nlp)
    @test neval_obj(nlp) == 0 && pass_counts(nlp).forward == 0 && pass_counts(nlp).backward == 0
  end
end
