@testset "minibatches" begin
  X = reshape(collect(1.0:20.0), 2, 10)             # 10 observations along the last dimension
  Y = zeros(3, 10)
  for i in 1:10
    Y[mod1(i, 3), i] = 1
  end
  make(; kwargs...) = LuxNLPModel(Dense(2 => 3), (X, Y); T = Float64, batchsize = 4, kwargs...)

  @testset "passes, epochs, reset" begin
    nlp = make(; shuffle = false)
    @test nlp.batch[1] == X[:, 1:4] && epoch(nlp) == 0
    @test minibatch_next_train!(nlp)
    @test nlp.batch[1] == X[:, 5:8]
    @test !minibatch_next_train!(nlp)                # 9:10 incomplete: dropped, new pass
    @test epoch(nlp) == 1 && nlp.batch[1] == X[:, 1:4]
    minibatch_next_train!(nlp)
    reset_minibatch_train!(nlp)
    @test epoch(nlp) == 1 && nlp.batch[1] == X[:, 1:4]
  end

  @testset "shuffled passes visit distinct observations" begin
    nlp = make(; batchsize = 5, seed = 1)
    seen = Float64[]
    for _ in 1:2
      append!(seen, nlp.batch[1][1, :])
      minibatch_next_train!(nlp)
    end
    @test sort(seen) == X[1, :]
  end

  @testset "the model's batch is its own copy" begin
    nlp = make(; shuffle = false)
    b = copy(nlp.batch[1])
    _, s = iterate(nlp.loader)
    iterate(nlp.loader, s)                           # loader buffer now holds observations 5:8
    @test nlp.batch[1] == b
  end

  @testset "set_data_train!" begin
    nlp = make(; shuffle = false)
    set_data_train!(nlp, (2 .* X, Y))
    @test epoch(nlp) == 0 && nlp.batch[1] == 2 .* X[:, 1:4]
    @test_throws ArgumentError set_data_train!(nlp, (Float32.(X), Float32.(Y)))   # other type
    @test_throws DimensionMismatch set_data_train!(nlp, (zeros(3, 10), Y))        # other shape
    @test_throws ArgumentError set_data_train!(nlp, (X[:, 1:3], Y[:, 1:3]))       # too few
  end

  @testset "constructor checks" begin
    @test_throws ArgumentError make(; batchsize = 11)
    @test_throws ArgumentError LuxNLPModel(Dense(2 => 3), (Float32.(X), Float32.(Y)); T = Float64)
  end

  @testset "NamedTuple data and a custom loss" begin
    data = (obs = randn(Xoshiro(3), 3, 32), target = randn(Xoshiro(4), 1, 32))
    function nt_loss(model, ps, st, b)
      ŷ, st_new = model(b.obs, ps, st)
      return sum(abs2, ŷ .- b.target) / size(ŷ, 2), st_new, (;)
    end
    nlp = LuxNLPModel(Dense(3 => 1), data; loss_f = nt_loss, T = Float64, batchsize = 8)
    x = nlp.meta.x0
    v = randn(Xoshiro(5), length(x))
    @test dot(grad(nlp, x), v) ≈ fd_dir(z -> obj(nlp, z), x, v) rtol = 1e-6
  end
end
