@testset "Hessian-vector products" begin
  @testset "backend $(backend isa Symbol ? backend : nameof(typeof(backend)))" for backend in
                                                                               (:finite_difference, SecondOrder(AutoForwardDiff(), AutoZygote()))
    nlp = toy_nlp(Float64; hvp_backend = backend)
    x = nlp.meta.x0
    rng = Xoshiro(3)
    v, w = randn(rng, length(x)), randn(rng, length(x))
    Hv, Hw = hprod(nlp, x, v), hprod(nlp, x, w)
    @test dot(w, Hv) ≈ dot(v, Hw) rtol = 1e-5                # symmetry
    @test Hv ≈ fd_dir(z -> grad(nlp, z), x, v) rtol = 1e-4   # agrees with differenced gradients
    @test hprod(nlp, x, v; obj_weight = 2.0) ≈ 2 .* Hv
    @test hprod(nlp, x, zero(v)) == zero(v)
  end

  @testset "switching backends resets the preparation" begin
    nlp = toy_nlp(Float64)
    x = nlp.meta.x0
    v = ones(length(x))
    Hv = hprod(nlp, x, v)                                    # prepares forward-over-reverse
    set_hvp_backend!(nlp, :finite_difference)
    @test nlp.prep_hvp === nothing
    @test hprod(nlp, x, v) ≈ Hv rtol = 1e-4
  end
end
