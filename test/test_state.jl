@testset "layer state" begin
  @testset "BatchNorm: precision and commit" begin
    X, Y = toy_data(Float64, Xoshiro(4))
    m = Chain(Dense(3 => 4), BatchNorm(4), Dense(4 => 2))
    nlp = LuxNLPModel(m, ArrayBatchSource((X, Y); batchsize = 16); T = Float64)
    x = nlp.meta.x0
    @test eltype(nlp.st.layer_2.running_mean) == Float64   # state converted to T
    _, st_new, _ = nlp.objective(nlp.model, params_view(nlp, x), nlp.st, current_batch(nlp))
    @test typeof(st_new) == typeof(nlp.st)                   # no conversion when stored

    rm0 = copy(nlp.st.layer_2.running_mean)
    f1 = obj(nlp, x)
    @test nlp.st.layer_2.running_mean == rm0                 # obj does not commit
    commit_state!(nlp)
    @test nlp.st.layer_2.running_mean != rm0                 # commit adopts the new statistics
    @test obj(nlp, x) ≈ f1 rtol = 1e-12                      # training-mode f does not use them
  end

  @testset "recalibration uses the nominal batch size and leaves f unchanged" begin
    X, Y = toy_data(Float64, Xoshiro(5))
    sizes = Int[]
    spy(model, ps, st, batch) = (push!(sizes, size(batch[2], 2)); classification_objective(model, ps, st, batch))
    m = Chain(Dense(3 => 4), BatchNorm(4), Dense(4 => 2))
    nlp = LuxNLPModel(m, ArrayBatchSource((X, Y); batchsize = 16); objective = spy, T = Float64)
    x = nlp.meta.x0
    use_full_batch!(nlp)
    f = obj(nlp, x)
    rm0 = copy(nlp.st.layer_2.running_mean)
    empty!(sizes)
    recalibrate_state!(nlp, x; nbatches = 3)
    @test sizes == [16, 16, 16]                              # not the 64 of the current batch
    @test nlp.st.layer_2.running_mean != rm0
    @test obj(nlp, x) ≈ f rtol = 1e-12
  end

  @testset "Dropout: same mask within a batch, new mask per batch" begin
    X, Y = toy_data(Float64, Xoshiro(6))
    m = Chain(Dense(3 => 8, tanh), Dropout(0.5), Dense(8 => 2))
    nlp = LuxNLPModel(m, ArrayBatchSource((X, Y); batchsize = 16); T = Float64)
    x = nlp.meta.x0
    v = ones(length(x))
    @test obj(nlp, x) == obj(nlp, x)                          # same batch, same mask
    @test dot(grad(nlp, x), v) ≈ fd_dir(z -> obj(nlp, z), x, v) rtol = 1e-6   # f and g agree
    r0 = copy(nlp.st.layer_2.rng)
    commit_state!(nlp)
    next_batch!(nlp)
    r1 = copy(nlp.st.layer_2.rng)
    @test r1 != r0                                            # new batch, new mask
    recalibrate_state!(nlp, x; nbatches = 3)
    @test nlp.st.layer_2.rng == r1                            # evaluation leaves masks alone
  end
end
