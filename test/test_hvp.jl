@testset "Hessian-vector products" begin
  @testset "properties" begin
    nlp = toy_nlp(Float64)
    x = nlp.meta.x0
    rng = Xoshiro(7)
    v, w = randn(rng, length(x)), randn(rng, length(x))
    Hv, Hw = hprod(nlp, x, v), hprod(nlp, x, w)
    @test dot(w, Hv) ≈ dot(v, Hw) rtol = 1e-5                # symmetry
    @test hprod(nlp, x, v; obj_weight = 2.0) ≈ 2 .* Hv
    @test hprod(nlp, x, v; obj_weight = 0.0) == zero(v)
    @test hprod(nlp, x, zero(v)) == zero(v)
  end
  @testset "Float64 obj_weight keeps Float32" begin
    nlp = toy_nlp(Float32)
    x = nlp.meta.x0
    @test eltype(hprod(nlp, x, ones(Float32, length(x)); obj_weight = 2.0)) == Float32
  end
  @testset "exact on least squares" begin
    X, y = randn(Xoshiro(8), 3, 32), randn(Xoshiro(9), 1, 32)
    nlp = LuxNLPModel(Dense(3 => 1), (X, y); loss_f = MSELoss(), T = Float64, batchsize = 32)
    A = [X; ones(1, 32)]                 # x = [weight; bias], f = ‖Aᵀx - yᵀ‖² / 32, H = 2AAᵀ / 32
    v = randn(Xoshiro(10), 4)
    @test hprod(nlp, nlp.meta.x0, v) ≈ 2 * A * (A' * v) / 32 rtol = 1e-6
    exact = LuxNLPModel(Dense(3 => 1), (X, y); loss_f = MSELoss(), T = Float64, batchsize = 32,
                        hvp_backend = EXACT_HVP)
    @test hprod(exact, exact.meta.x0, v) ≈ 2 * A * (A' * v) / 32 rtol = 1e-12
    @test isempty(exact.w)                                   # no finite-difference workspace
  end
  @testset "exact backend agrees with finite differences" begin
    fd, exact = toy_nlp(Float64), toy_nlp(Float64; hvp_backend = EXACT_HVP)
    x = fd.meta.x0
    v = randn(Xoshiro(11), length(x))
    @test hprod(exact, x, v) ≈ hprod(fd, x, v) rtol = 1e-6
    @test hprod(exact, x, v; obj_weight = 3.0) ≈ 3 .* hprod(exact, x, v)
    minibatch_next_train!(fd)
    minibatch_next_train!(exact)
    @test hprod(exact, x, v) ≈ hprod(fd, x, v) rtol = 1e-6  # follows the minibatch
    set_batchsize!(fd, 32)
    set_batchsize!(exact, 32)
    @test hprod(exact, x, v) ≈ hprod(fd, x, v) rtol = 1e-6  # and the batch size
  end
end
