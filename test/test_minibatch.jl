@testset "minibatches" begin
  X = reshape(collect(1.0:20.0), 2, 10)             # 10 observations along the last dimension
  Y = zeros(3, 10)
  for i in 1:10
    Y[mod1(i, 3), i] = 1
  end
  make(; kwargs...) = LuxNLPModel(Dense(2 => 3), (X, Y); T = Float64, batchsize = 4, kwargs...)

  @testset "passes and epochs" begin
    nlp = make(; shuffle = false)
    @test nlp.batch[1] == X[:, 1:4] && epoch(nlp) == 0
    @test minibatch_next_train!(nlp)
    @test nlp.batch[1] == X[:, 5:8]
    @test !minibatch_next_train!(nlp)                # 9:10 incomplete: dropped, new pass
    @test epoch(nlp) == 1 && nlp.batch[1] == X[:, 1:4]
  end

  @testset "set_batchsize!" begin
    nlp = make(; shuffle = false)
    minibatch_next_train!(nlp)
    @test set_batchsize!(nlp, 5) === nlp
    @test nlp.batch[1] == X[:, 1:5] && epoch(nlp) == 0     # the pass restarts with the new size
    @test minibatch_next_train!(nlp) && nlp.batch[1] == X[:, 6:10]
    @test !minibatch_next_train!(nlp) && epoch(nlp) == 1
    x = nlp.meta.x0
    @test grad(nlp, x) ≈ grad(make(; shuffle = false, batchsize = 5), x)
    @test_throws ArgumentError set_batchsize!(nlp, 11)
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

  @testset "constructor" begin
    @test_throws ArgumentError make(; batchsize = 11)
    nlp = LuxNLPModel(Dense(2 => 3), (Float32.(X), Float32.(Y)); T = Float64, batchsize = 4)
    @test eltype(nlp.batch[1]) == Float64             # data converted to T
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
