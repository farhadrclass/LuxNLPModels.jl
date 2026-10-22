@testset "layer state" begin
  @testset "f and ∇f describe one function with $(nameof(typeof(layer)))" for layer in (BatchNorm(4), Dropout(0.5))
    m = Chain(Dense(3 => 4, tanh), layer, Dense(4 => 2))
    nlp = LuxNLPModel(m, toy_data(Float64, Xoshiro(8)); T = Float64, batchsize = 16)
    x = nlp.meta.x0
    v = randn(Xoshiro(9), length(x))
    for _ in 1:2
      @test obj(nlp, x) == obj(nlp, x)
      @test dot(grad(nlp, x), v) ≈ fd_dir(z -> obj(nlp, z), x, v) rtol = 1e-6
      minibatch_next_train!(nlp)
    end
  end

  @testset "BatchNorm state has the precision T" begin
    m = Chain(Dense(3 => 4), BatchNorm(4), Dense(4 => 2))
    nlp = LuxNLPModel(m, toy_data(Float64, Xoshiro(10)); T = Float64, batchsize = 16)
    @test eltype(nlp.st.layer_2.running_mean) == Float64
  end

  @testset "Dropout: one mask per minibatch" begin
    m = Chain(Dense(3 => 8, tanh), Dropout(0.5), Dense(8 => 2))
    nlp = LuxNLPModel(m, toy_data(Float64, Xoshiro(11)); T = Float64, batchsize = 64, shuffle = false)
    x = nlp.meta.x0
    f = obj(nlp, x)
    r = copy(nlp.st.layer_2.rng)
    @test !minibatch_next_train!(nlp)                # one batch per pass: same 64 observations
    @test nlp.st.layer_2.rng != r                    # but a new mask
    @test obj(nlp, x) != f
  end

  @testset "set_batchsize! also draws new Dropout masks" begin
    m = Chain(Dense(3 => 8, tanh), Dropout(0.5), Dense(8 => 2))
    nlp = LuxNLPModel(m, toy_data(Float64, Xoshiro(12)); T = Float64, batchsize = 16)
    r = copy(nlp.st.layer_2.rng)
    set_batchsize!(nlp, 32)
    @test nlp.st.layer_2.rng != r
    x = nlp.meta.x0
    @test obj(nlp, x) == obj(nlp, x)
  end
end
