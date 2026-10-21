@testset "Hessian-vector products" begin
  @testset "properties" begin
    nlp = toy_nlp(Float64)
    x = nlp.meta.x0
    rng = Xoshiro(7)
    v, w = randn(rng, length(x)), randn(rng, length(x))
    Hv, Hw = hprod(nlp, x, v), hprod(nlp, x, w)
    @test dot(w, Hv) ≈ dot(v, Hw) rtol = 1e-5                # symmetry
    @test hprod(nlp, x, v; obj_weight = 2.0) ≈ 2 .* Hv
    @test hprod(nlp, x, zero(v)) == zero(v)
  end
  @testset "exact on least squares" begin
    X, y = randn(Xoshiro(8), 3, 32), randn(Xoshiro(9), 1, 32)
    nlp = LuxNLPModel(Dense(3 => 1), (X, y); loss_f = MSELoss(), T = Float64, batchsize = 32)
    A = [X; ones(1, 32)]                 # x = [weight; bias], f = ‖Aᵀx - yᵀ‖² / 32, H = 2AAᵀ / 32
    v = randn(Xoshiro(10), 4)
    @test hprod(nlp, nlp.meta.x0, v) ≈ 2 * A * (A' * v) / 32 rtol = 1e-6
  end
end
