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

  @testset "evaluation recomputes BatchNorm statistics and leaves nlp.st alone" begin
    m = Chain(Dense(3 => 4), BatchNorm(4), Dense(4 => 2))
    X, Y = toy_data(Float64, Xoshiro(12); n = 256)
    X .+= 3                                          # nonzero mean: the initial statistics are wrong
    nlp = LuxNLPModel(m, (X, Y), (X, Y); T = Float64, batchsize = 32)
    x = nlp.meta.x0
    st0 = deepcopy(nlp.st)
    f = obj(nlp, x)
    st = LuxNLPModels._testmode_state(nlp, params_view(nlp, x))
    @test nlp.st == st0 && obj(nlp, x) == f          # solver-side state and f unchanged
    h, _ = nlp.model.layers.layer_1(X, params_view(nlp, x).layer_1, nlp.st.layer_1)
    μ = vec(mean(h; dims = 2))
    @test norm(st.layer_2.running_mean - μ) < 0.25 * norm(μ)
    @test norm(nlp.st.layer_2.running_mean - μ) > 0.9 * norm(μ)
    a = accuracy(nlp, x)
    @test 0 <= a <= 1 && nlp.st == st0
  end

  @testset "accuracy matches a direct computation" begin
    nlp = toy_nlp(Float64)
    x = nlp.meta.x0
    X, Y = nlp.data_test
    ŷ, _ = nlp.model(X, params_view(nlp, x), Lux.testmode(nlp.st))
    direct = mean(argmax(ŷ[:, j]) == argmax(Y[:, j]) for j in axes(Y, 2))
    @test accuracy(nlp, x) ≈ direct
  end
end
