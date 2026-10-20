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
    U = T === Float32 ? Float64 : Float32
    @test_throws ArgumentError obj(nlp, ones(U, length(x)))     # precision fixed at construction
  end

  @testset "gradients across minibatches" begin
    nlp = toy_nlp(Float64)
    x = copy(nlp.meta.x0)
    v = randn(Xoshiro(6), length(x))
    for _ in 1:5                                     # crosses an epoch boundary (4 batches/pass)
      g = grad(nlp, x)
      @test dot(g, v) ≈ fd_dir(z -> obj(nlp, z), x, v) rtol = 1e-6
      fx, g2 = objgrad!(nlp, x, similar(x))
      @test fx ≈ obj(nlp, x)
      @test g2 ≈ g
      minibatch_next_train!(nlp)
    end
    @test epoch(nlp) == 1
  end

  @testset "counters" begin
    nlp = toy_nlp(Float64)
    x = nlp.meta.x0
    objgrad!(nlp, x, similar(x))
    hprod(nlp, x, ones(length(x)))
    @test neval_obj(nlp) == 1 && neval_grad(nlp) == 1 && neval_hprod(nlp) == 1
    NLPModels.reset!(nlp)
    @test neval_obj(nlp) == 0
  end
end
